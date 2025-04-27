from __future__ import absolute_import, unicode_literals

import os
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import joblib
from celery import shared_task
from django.core.mail import send_mail
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

from .models import WhatIfTaskResult

# Setup cache directory
CACHE_DIR = os.path.join(os.getcwd(), "cached_models")
os.makedirs(CACHE_DIR, exist_ok=True)

# Utility Functions
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def predict_future_prices(model, data, n_days, seq_len, scaler):
    last_sequence = data[-seq_len:]
    future_predictions = []
    for _ in range(n_days):
        input_seq = last_sequence.reshape(1, seq_len, 1)
        next_price = model.predict(input_seq, verbose=0)[0][0]
        future_predictions.append(next_price)
        last_sequence = np.append(last_sequence[1:], [[next_price]], axis=0)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

def build_model(volatility_category):
    model = Sequential()
    if volatility_category == "low":
        model.add(LSTM(64, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32))
    elif volatility_category == "high":
        model.add(LSTM(50, input_shape=(60, 1)))
    else:
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@shared_task(bind=True)
def what_if_background_analysis(self, user_email, symbol, investment_date_str, end_date_str, investment_amount):
    task_id = self.request.id
    try:
        # Create the task record
        task_obj = WhatIfTaskResult.objects.create(
            task_id=task_id,
            user_email=user_email,
            status="pending",
        )

        set_seed()

        today = datetime.date.today()
        investment_date = datetime.datetime.strptime(investment_date_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

        # Step 1: Historical data
        stock = yf.Ticker(symbol)
        hist_data = stock.history(start=investment_date - datetime.timedelta(days=1), end=today + datetime.timedelta(days=1))
        hist_data.index = hist_data.index.date

        if hist_data.empty:
            raise ValueError("No historical data found.")
        if investment_date not in hist_data.index:
            raise ValueError("No data available for the investment date. Market might have been closed.")

        investment_price = hist_data.loc[investment_date]['Open']
        if np.isnan(investment_price):
            investment_price = hist_data.loc[investment_date]['Close']

        historical_dates = [d.strftime("%Y-%m-%d") for d in hist_data.index if d >= investment_date]
        historical_prices = [float(p) for i, p in enumerate(hist_data['Close']) if hist_data.index[i] >= investment_date]

        predicted_dates = []
        predicted_prices = []

        # Step 2: Future prediction if needed
        if end_date > today:
            future_days = (end_date - today).days

            cache_model_path = os.path.join(CACHE_DIR, f"{symbol}_{today}_model.h5")
            cache_scaler_path = os.path.join(CACHE_DIR, f"{symbol}_{today}_scaler.pkl")

            if os.path.exists(cache_model_path) and os.path.exists(cache_scaler_path):
                model = load_model(cache_model_path)
                scaler = joblib.load(cache_scaler_path)
            else:
                df_vol = yf.download(symbol, period="1y")
                returns = df_vol['Close'].pct_change()
                vol_score = float(returns.std())
                vol_category = "low" if vol_score < 0.015 else "high" if vol_score > 0.03 else "medium"

                df = yf.download(symbol, period="10y")
                df = df[['Close']].dropna()

                if df.empty:
                    raise ValueError(f"No historical data available for {symbol}.")

                scaler = MinMaxScaler()
                scaled_close = scaler.fit_transform(df[['Close']].values)
                X, y = create_sequences(scaled_close)
                split = int(0.8 * len(X))
                X_train, y_train = X[:split], y[:split]

                model = build_model(vol_category)
                model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0,
                          callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)])

                model.save(cache_model_path)
                joblib.dump(scaler, cache_scaler_path)

            df = yf.download(symbol, period="10y")
            df = df[['Close']].dropna()
            scaled_close = scaler.transform(df[['Close']].values)
            future_prices = predict_future_prices(model, scaled_close, future_days, 60, scaler).flatten()

            last_real_price = df['Close'].iloc[-1].item()
            adjustment = last_real_price - future_prices[0]
            future_prices += adjustment

            predicted_dates = [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, len(future_prices) + 1)]
            predicted_prices = [float(p) for p in future_prices.tolist()]

            final_price = predicted_prices[-1]
        else:
            final_price = historical_prices[-1]

        # Step 3: Final calculations
        final_value = (float(investment_amount) / investment_price) * final_price
        percentage_change = ((final_value - float(investment_amount)) / float(investment_amount)) * 100

        result_data = {
            "symbol": symbol,
            "investment_date": str(investment_date),
            "end_date": str(end_date),
            "initial_price": float(round(investment_price, 2)),
            "final_price": float(round(final_price, 2)),
            "investment_amount": float(round(float(investment_amount), 2)),
            "final_value": float(round(final_value, 2)),
            "percentage_change": float(round(percentage_change, 2)),
            "historical_dates": historical_dates,
            "historical_prices": historical_prices,
            "predicted_dates": predicted_dates,
            "predicted_prices": predicted_prices,
        }

        task_obj.status = "completed"
        task_obj.result_json = result_data
        task_obj.save()

        # Step 4: Send the email
        result_url = f"http://127.0.0.1:8000/what-if-result/?task_id={task_id}"

        email_body = f"""
        Hello!

        Your What-If Investment Analysis for {symbol} is ready:

        Investment Date: {investment_date}
        End Date: {end_date}
        Investment Amount: £{investment_amount}
        Final Value: £{final_value:.2f}
        Percentage Change: {percentage_change:.2f}%

        👉 View your full result here:
        {result_url}

        Regards,
        InvestGrow Team
        """

        try:
            send_mail(
                subject=f"Your What-If Analysis for {symbol}",
                message=email_body,
                from_email=None,  # Configure your actual email in settings
                recipient_list=[user_email],
                fail_silently=False,
            )
        except Exception as e:
            print("Email sending failed:", str(e))

    except Exception as e:
        WhatIfTaskResult.objects.filter(task_id=task_id).update(
            status="failed",
            error_message=str(e),
        )
        try:
            send_mail(
                subject="Analysis Failed",
                message=f"An error occurred during your What-If Analysis: {str(e)}",
                from_email=None,
                recipient_list=[user_email],
                fail_silently=True,
            )
        except:
            pass