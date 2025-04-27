import os
import random
import matplotlib
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, logger
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.metrics import r2_score, mean_squared_error
from .forms import CustomUserCreationForm
import requests
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from django.shortcuts import get_object_or_404
from .models import *
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
import plotly.graph_objs as go
from django.shortcuts import redirect, render
from allauth.socialaccount.models import SocialAccount
from allauth.account.models import EmailAddress
from .models import UserQuizProgress
from django.views.decorators.http import require_GET, require_POST
from .models import WatchedStock
import json
from django.views.decorators.http import require_GET
from django.core.cache import cache
import datetime
import base64
from io import BytesIO
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
matplotlib.use('Agg')
from Financial_Investment_Assistant.tasks import what_if_background_analysis
import tensorflow as tf

CACHE_DIR = os.path.join(os.getcwd(), "cached_models")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def landing_page(request):
    return render(request, 'landing.html')

def help_page(request):
    return render(request, "help.html")

def stats_api(request):
    total_users = User.objects.count()
    total_xp = UserQuizProgress.objects.aggregate(total_xp_sum=models.Sum('total_xp'))['total_xp_sum'] or 0
    return JsonResponse({
        "users": total_users,
        "total_xp": total_xp,
    })
@login_required(login_url='/accounts/login/')
def profile_view(request):
    user = request.user
    progress = UserQuizProgress.objects.filter(user=user).first()
    email = EmailAddress.objects.filter(user=user, primary=True).first()
    social_accounts = SocialAccount.objects.filter(user=user)

    context = {
        "user": user,
        "email": email,
        "progress": progress,
        "social_accounts": social_accounts
    }
    return render(request, "account/profile.html", context)

def fetch_stock_search(request):
    query = request.GET.get("query", "").strip()
    if not query:
        return JsonResponse({"error": "No search query provided"}, status=400)

    cache_key = f"stock_search_{query.lower()}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return JsonResponse({"bestMatches": cached_result})

    # Yahoo Finance autocomplete API
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return JsonResponse({"error": "Failed to fetch data"}, status=500)

        data = response.json()
        results = []
        for stock in data.get("quotes", []):  # Extract relevant stock information
            results.append({
                "symbol": stock.get("symbol", "N/A"),
                "name": stock.get("shortname", stock.get("longname", "Unknown")),
                "exchange": stock.get("exchange", "Unknown"),
                "type": stock.get("quoteType", "Unknown")
            })

        cache.set(cache_key, results, timeout=60 * 60 * 6)
        return JsonResponse({"bestMatches": results})

    except Exception as e:
        return JsonResponse({"error": f"Failed to fetch data: {str(e)}"}, status=500)

def get_fundamentals(symbol):
    cache_key = f"fundamentals_{symbol.upper()}_{datetime.date.today()}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data

    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        fundamentals = {
            "market_cap": f"${info.get('marketCap', 'N/A'):,}",
            "pe_ratio": info.get("trailingPE", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "dividend_yield": f"{round(info.get('dividendYield', 0) * 100, 2)}%" if info.get("dividendYield") else "N/A",
            "sector": info.get("sector", "N/A"),
            "country": info.get("country", "N/A"),
        }

        cache.set(cache_key, fundamentals, timeout=60 * 60 * 12)
        return fundamentals

    except Exception as e:
        print("Error fetching fundamentals:", e)
        return None

def fetch_gainers_losers(request):
    cache_key = f"gainers_losers_{datetime.date.today()}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return JsonResponse(cached_result)

    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        gainers_res = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_gainers&count=5",
            headers=headers
        )
        losers_res = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_losers&count=5",
            headers=headers
        )

        gainers_data = gainers_res.json()
        losers_data = losers_res.json()

        result = {
            "gainers": gainers_data["finance"]["result"][0].get("quotes", []),
            "losers": losers_data["finance"]["result"][0].get("quotes", [])
        }

        cache.set(cache_key, result, timeout=60 * 60 * 6)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_GET
def fetch_market_summary(request):
    indices = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq",
        "^DJI": "Dow Jones",
        "^RUT": "Russell 2000",
        "^VIX": "VIX",
        "^N225": "Nikkei 225",
    }

    summary = []

    cache_key = f"market_summary_{datetime.date.today()}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return JsonResponse({"summary": cached_data})

    try:
        for symbol, name in indices.items():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current = info.get("regularMarketPrice")
            previous = info.get("previousClose")

            if current is not None and previous is not None:
                change = current - previous
                percent_change = (change / previous) * 100 if previous else 0

                summary.append({
                    "name": name,
                    "price": round(current, 2),
                    "change": round(change, 2),
                    "percent_change": round(percent_change, 2),
                    "is_up": change >= 0
                })

        cache.set(cache_key, summary, timeout=60 * 60 * 4)
        return JsonResponse({"summary": summary})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_GET
def fetch_market_chart_data(request):
    symbol = request.GET.get("symbol")

    if not symbol:
        return JsonResponse({"error": "Missing symbol"}, status=400)

    cache_key = f"market_chart_{symbol.upper()}_{datetime.date.today()}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return JsonResponse(cached_data)

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", interval="1d")

        if hist.empty:
            return JsonResponse({"error": "No data available"}, status=404)

        closing_prices = hist["Close"].tolist()
        result = {"symbol": symbol, "prices": closing_prices}

        cache.set(cache_key, result, timeout=60 * 60 * 6)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_POST
@csrf_exempt
@login_required
def add_to_watchlist(request):
    try:
        data = json.loads(request.body)
        symbol = data.get("symbol", "").upper()

        if not symbol:
            return JsonResponse({"error": "Symbol is required"}, status=400)

        # Check if already exists
        exists = WatchedStock.objects.filter(user=request.user, symbol=symbol).exists()
        if exists:
            return JsonResponse({"error": f"{symbol} is already in your watchlist."}, status=400)

        WatchedStock.objects.create(user=request.user, symbol=symbol)
        return JsonResponse({"success": True})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_POST
@login_required
def delete_stock(request):
    symbol = request.POST.get("symbol")

    if not symbol:
        return JsonResponse({"success": False, "error": "No symbol provided."})

    try:
        stock = WatchedStock.objects.get(symbol=symbol.upper(), user=request.user)
        stock.delete()
        return JsonResponse({"success": True})
    except WatchedStock.DoesNotExist:
        return JsonResponse({"success": False, "error": "Stock not found."})
@login_required(login_url='/accounts/login/')
def index(request):
    stock_symbol = request.GET.get('symbol', 'AAPL')
    time_range = request.GET.get('time_range', 'daily').lower()

    chart_html = None
    error_message = None
    fundamentals = get_fundamentals(stock_symbol)

    # Mapping time_range to yfinance parameters
    period_interval_map = {
        'daily': ('1mo', '1d'),
        'weekly': ('3mo', '1wk'),
        'monthly': ('1y', '1mo')
    }
    period, interval = period_interval_map.get(time_range, ('1mo', '1d'))

    try:
        ticker = yf.Ticker(stock_symbol)
        hist = ticker.history(period=period, interval=interval)

        if not hist.empty:
            close_prices = hist["Close"].tail(30)
            dates = close_prices.index.strftime('%Y-%m-%d')

            trace = go.Scatter(
                x=dates,
                y=close_prices.values,
                mode='lines+markers',
                name=f'{stock_symbol} Closing Prices',
                line=dict(color='blue')
            )

            layout = go.Layout(
                title=f"{stock_symbol} - {time_range.capitalize()} Closing Prices",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Price (USD)"),
                template="plotly_dark",
                showlegend=False
            )

            fig = go.Figure(data=[trace], layout=layout)
            chart_html = fig.to_html(full_html=False)
        else:
            error_message = f"No data available for '{stock_symbol}'."
    except Exception as e:
        error_message = f"Error fetching data for '{stock_symbol}': {str(e)}"

    # Watchlist
    watchlist = WatchedStock.objects.filter(user=request.user)
    watched_stocks_data = []

    for item in watchlist:
        try:
            ticker = yf.Ticker(item.symbol)
            info = ticker.info
            watched_stocks_data.append({
                "symbol": item.symbol,
                "name": info.get("shortName"),
                "price": info.get("regularMarketPrice"),
                "change": info.get("regularMarketChange"),
                "percent": info.get("regularMarketChangePercent"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
            })
        except:
            continue

    sector_stocks = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMD", "NVDA"],
        "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "UNH"],
        "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "F"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Industrials": ["BA", "GE", "CAT", "UPS", "MMM"],
        "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
        "Communication Services": ["NFLX", "DIS", "T", "VZ", "CHTR"],
        "Basic Materials": ["LIN", "SHW", "NEM", "FCX", "DD"],
        "Utilities": ["NEE", "DUK", "SO", "AEP", "EXC"],
        "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "SPG"],
    }

    # Suggested Stocks
    suggested_stocks = []
    if fundamentals:
        sector = fundamentals.get('sector')
        if sector and sector in sector_stocks:
            symbols_to_suggest = sector_stocks[sector]
            for sym in symbols_to_suggest:
                if not WatchedStock.objects.filter(user=request.user, symbol=sym).exists():
                    try:
                        tick = yf.Ticker(sym)
                        info = tick.info
                        suggested_stocks.append({
                            "symbol": sym,
                            "name": info.get("shortName"),
                            "price": info.get("regularMarketPrice"),
                            "change": info.get("regularMarketChange"),
                            "percent": info.get("regularMarketChangePercent"),
                            "market_cap": info.get("marketCap"),
                            "pe_ratio": info.get("trailingPE"),
                        })
                    except:
                        continue

    return render(request, 'index.html', {
        'chart_html': chart_html,
        'stock_symbol': stock_symbol,
        'time_range': time_range,
        'error_message': error_message,
        'fundamentals': fundamentals,
        'watched_stocks': watched_stocks_data,
        'suggested_stocks': suggested_stocks,
    })

def fetch_news(request):
    news_api_key = "d781a3bd28f946c78951088c88aa9f45"
    query = request.GET.get("query", "").strip()  # Get user input query
    articles = []

    # Fetch finance-related news by default
    base_url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={news_api_key}"

    # If user searched for a specific keyword, fetch relevant news first
    if query:
        search_url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=10&apiKey={news_api_key}"
        response = requests.get(search_url).json()
        articles = response.get("articles", [])

    # If the search results are less than 10, fill up with general finance news
    if len(articles) < 10:
        business_news = requests.get(base_url).json()
        articles.extend(business_news.get("articles", []))  # Add general news
        articles = articles[:10]  # Ensure only 10 articles are returned

    return JsonResponse({"articles": articles})

def fetch_trending_stocks(request):
    cache_key = f"trending_stocks_{datetime.date.today()}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return JsonResponse({"trending": cached_data})

    try:
        # Fetch trending stock symbols from Yahoo Finance API
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = response.json()

        print("Raw API Response:", data)  # Debugging

        # Check if the response contains the expected data
        if "finance" in data and "result" in data["finance"] and len(data["finance"]["result"]) > 0:
            trending_stocks = []
            symbols = [stock["symbol"] for stock in data["finance"]["result"][0]["quotes"]]

            # Fetch stock prices using Yahoo Finance (yfinance)
            stock_data = yf.Tickers(" ".join(symbols))

            for symbol in symbols:
                try:
                    stock_info = stock_data.tickers[symbol].history(period="1d")
                    latest_price = round(stock_info['Close'].iloc[-1], 2) if not stock_info.empty else "N/A"
                except:
                    latest_price = "N/A"

                trending_stocks.append({
                    "symbol": symbol,
                    "price": latest_price
                })
            cache.set(cache_key, trending_stocks[:14], timeout=60 * 60 * 3)
            return JsonResponse({"trending": trending_stocks[:14]})

        return JsonResponse({"error": "Unexpected API structure"}, status=500)

    except Exception as e:
        print("Error fetching trending stocks:", str(e))  # Debugging
        return JsonResponse({"error": str(e)}, status=500)

def signup_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = True  # Skip email verification for now
            user.save()
            return redirect('login')  # Redirect to Login or home page
        else:
            print("Form Errors:", form.errors)
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

# Calculate Exponential Moving Average
def calculate_ema(data, period):
    ema = data['Close'].ewm(span=period, adjust=False).mean()
    return ema

# Calculate Relative Strength Index
def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Generate Buy/Sell signals with Risk Adjustment
def generate_signals(data, risk_level):
    data['Signal'] = 0

    # print("RSI Values:", data['RSI'].tail())  # Print last few RSI values
    # print("EMA Values:", data[['EMA_5', 'EMA_8', 'EMA_13']].tail())  # Print last few EMA values

    if risk_level == 'low':  # Conservative signals
        data.loc[
            (data['RSI'] < 30) &
            (data['EMA_5'] > data['EMA_8']) &
            (data['EMA_8'] > data['EMA_13']),
            'Signal'
        ] = 1  # Buy
        data.loc[
            (data['RSI'] > 70) &
            (data['EMA_5'] < data['EMA_8']) &
            (data['EMA_8'] < data['EMA_13']),
            'Signal'
        ] = -1  # Sell
    elif risk_level == 'moderate':  # Default signals
        data.loc[
            (data['RSI'] < 40) &
            (data['EMA_5'] > data['EMA_8']) &
            (data['EMA_8'] > data['EMA_13']),
            'Signal'
        ] = 1  # Buy
        data.loc[
            (data['RSI'] > 60) &
            (data['EMA_5'] < data['EMA_8']) &
            (data['EMA_8'] < data['EMA_13']),
            'Signal'
        ] = -1  # Sell
    elif risk_level == 'high':  # Aggressive signals
        data.loc[
            (data['RSI'] < 50) &
            (data['EMA_5'] > data['EMA_8']) &
            (data['EMA_8'] > data['EMA_13']),
            'Signal'
        ] = 1  # Buy
        data.loc[
            (data['RSI'] > 50) &
            (data['EMA_5'] < data['EMA_8']) &
            (data['EMA_8'] < data['EMA_13']),
            'Signal'
        ] = -1  # Sell
    return data

# Plot candlestick chart with signals
def plot_candlestick(data):
    data['Date'] = mdates.date2num(data.index)
    ohlc_data = data[['Date', 'Open', 'High', 'Low', 'Close']]

    plt.figure(figsize=(14, 7))
    ax = plt.subplot()

    candlestick_ohlc(ax, ohlc_data.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

    plt.plot(data.index, data['EMA_5'], label="EMA 5", color='blue', linestyle='--', linewidth=1)
    plt.plot(data.index, data['EMA_8'], label="EMA 8", color='orange', linestyle='--', linewidth=1)
    plt.plot(data.index, data['EMA_13'], label="EMA 13", color='purple', linestyle='--', linewidth=1)

    # Plot buy and sell signals
    plt.scatter(data.index[data['Signal'] == 1], data['Close'][data['Signal'] == 1], label="Buy Signal", marker='^', color='green', alpha=1)
    plt.scatter(data.index[data['Signal'] == -1], data['Close'][data['Signal'] == -1], label="Sell Signal", marker='v', color='red', alpha=1)

    plt.title("Candlestick Chart with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    # Convert plot to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return chart_image

@login_required(login_url='/accounts/login/')
def buy_sell(request):
    symbol = request.GET.get('symbol', 'AAPL')
    risk_level = request.GET.get('risk_level', 'moderate').strip()

    recommendation = chart_html = error_message = model_metrics = None
    feature_importance = []
    current_rsi = current_volatility = price_ema_ratio = predicted_return = confidence = None

    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="3y")
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Feature Engineering
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Close'].rolling(20).std()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

        for window in [5, 12, 20, 26, 50]:
            data[f'SMA_{window}'] = data['Close'].rolling(window).mean()
            data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()

        def calculate_rsi(data, period=14):
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        data['RSI'] = calculate_rsi(data)
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['Volume_MA'] = data['Volume'].rolling(5).mean()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Upper_Band'] = data['SMA_20'] + 2 * data['Close'].rolling(20).std()
        data['Lower_Band'] = data['SMA_20'] - 2 * data['Close'].rolling(20).std()
        data['Future_Return'] = data['Close'].shift(-5) / data['Close'] - 1
        data.dropna(inplace=True)

        features = ['Returns', 'Volatility', 'Log_Returns', 'RSI', 'MACD',
                    'Volume_MA', 'Volume_Change', 'SMA_5', 'SMA_20', 'SMA_50',
                    'EMA_5', 'EMA_20', 'EMA_50', 'Upper_Band', 'Lower_Band']
        X = data[features]
        y = data['Future_Return']

        tscv = TimeSeriesSplit(n_splits=5)
        r2_scores = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            split_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
            y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                n_jobs=-1,
                early_stopping_rounds=50
            )

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_test)
            r2_scores.append(r2_score(y_test, preds))

        avg_r2 = np.mean(r2_scores)
        model_metrics = f"Cross-validated R²: {min(r2_scores):.2f}-{max(r2_scores):.2f} (Avg: {avg_r2:.2f})"

        final_model = XGBRegressor(
            n_estimators=int(model.best_iteration * 1.1) if hasattr(model, 'best_iteration') else 200,
            learning_rate=0.01,
            max_depth=3,
            objective='reg:squarederror'
        )
        final_model.fit(X, y)

        latest_features = X.iloc[-1:]
        predicted_return = final_model.predict(latest_features)[0]

        current_rsi = data['RSI'].iloc[-1]
        current_volatility = data['Volatility'].iloc[-1]
        price_ema_ratio = data['Close'].iloc[-1] / data['EMA_50'].iloc[-1]

        risk_params = {
            'low': {'threshold': 0.015, 'rsi_max': 30},
            'moderate': {'threshold': 0.01, 'rsi_max': 40},
            'high': {'threshold': 0.005, 'rsi_max': 50}
        }
        params = risk_params.get(risk_level, risk_params['moderate'])
        volatility_adjustment = np.clip(current_volatility * 10, 0.5, 2)
        final_threshold = params['threshold'] * volatility_adjustment

        buy_signal = (predicted_return > final_threshold) and (current_rsi < params['rsi_max'])
        sell_signal = (predicted_return < -final_threshold) and (current_rsi > 70)

        confidence = 0
        if buy_signal:
            confidence = int(100 / (1 + np.exp(-10 * (predicted_return - final_threshold))))
        elif sell_signal:
            confidence = int(100 / (1 + np.exp(-10 * (-predicted_return - final_threshold))))

        if buy_signal:
            recommendation = f"""
            <strong>Recommendation: BUY {symbol}</strong> (Risk: {risk_level.capitalize()})<br>
            <strong>Predicted 5-day return:</strong> {predicted_return * 100:.2f}%<br>
            <strong>Buy threshold:</strong> {final_threshold * 100:.2f}%<br>
            <strong>Why?</strong>
            <ul>
                <li><strong>RSI:</strong> {current_rsi:.1f} — oversold zone</li>
                <li><strong>Volatility:</strong> {current_volatility:.2f} — acceptable risk</li>
                <li><strong>Price/EMA50:</strong> {price_ema_ratio:.2f} — price is near average</li>
                <li><strong>Model Confidence:</strong> {confidence:.1f}%</li>
            </ul>
            """
        elif sell_signal:
            recommendation = f"""
            🔻 <strong>Recommendation: SELL {symbol}</strong> (Risk: {risk_level.capitalize()})<br>
            📉 <strong>Predicted 5-day return:</strong> {predicted_return * 100:.2f}%<br>
            🎯 <strong>Sell threshold:</strong> {-final_threshold * 100:.2f}%<br>
            🧠 <strong>Why?</strong>
            <ul>
                <li><strong>RSI:</strong> {current_rsi:.1f} — overbought zone</li>
                <li><strong>Volatility:</strong> {current_volatility:.2f} — potential trend reversal</li>
                <li><strong>Price/EMA50:</strong> {price_ema_ratio:.2f} — above fair value</li>
                <li><strong>Model Confidence:</strong> {confidence:.1f}%</li>
            </ul>
            """
        else:
            recommendation = f"""
            🤔 <strong>Recommendation: HOLD {symbol}</strong> (Risk: {risk_level.capitalize()})<br>
            📊 <strong>Predicted 5-day return:</strong> {predicted_return * 100:.2f}%<br>
            🧠 <strong>Why?</strong>
            <ul>
                <li>Indicators do not show a strong buy or sell signal</li>
                <li>RSI: {current_rsi:.1f}</li>
                <li>Volatility: {current_volatility:.2f}</li>
                <li>Model Confidence: {confidence:.1f}%</li>
            </ul>
            """

        # === Simple Chart (Price + Buy Signals Only) ===
        predicted_returns = final_model.predict(X)
        buy_zones = (predicted_returns > final_threshold) & (data['RSI'] < params['rsi_max'])
        buy_dates = data.index[buy_zones]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='white')))
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=data.loc[buy_dates, 'Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))

        fig.update_layout(
            title=f"{symbol} Price Chart (Buy Signals Highlighted)",
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified'
        )
        chart_html = fig.to_html(full_html=False)

    except Exception as e:
        error_message = f"Error analyzing {symbol}: {str(e)}"
        logger.error(f"Error in buy_sell: {error_message}", exc_info=True)

    return render(request, 'buy_sell.html', {
        'chart_html': chart_html,
        'error_message': error_message,
        'recommendation': recommendation,
        'model_metrics': model_metrics,
        'symbol': symbol,
        'risk_level': risk_level,
        'feature_importance': feature_importance,
        'current_rsi': current_rsi,
        'current_volatility': current_volatility,
        'price_ema_ratio': price_ema_ratio,
        'predicted_return': predicted_return,
        'confidence': confidence,
    })

# Helpers
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

# Main View
@login_required(login_url='/accounts/login/')
def what_if_analysis(request):
    result = None
    error_message = None
    task_id = request.GET.get('task_id')  # <-- get task_id from URL if exists

    if request.method == 'GET' and 'symbol' in request.GET:
        symbol = request.GET.get('symbol', '').strip().upper()
        investment_date_str = request.GET.get('investment_date', '')
        end_date_str = request.GET.get('end_date', '')
        investment_amount = request.GET.get('investment_amount', '')

        try:
            investment_amount = float(investment_amount)
            today = datetime.date.today()
            investment_date = datetime.datetime.strptime(investment_date_str, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

            set_seed()

            if investment_date > today:
                raise ValueError("Investment date cannot be in the future.")
            elif end_date < investment_date:
                raise ValueError("End date must be after or equal to the investment date.")
            elif (end_date - today).days > 30:
                raise ValueError("End date must be within 30 days from today.")
            elif end_date > today and not task_id:
                # Start future prediction task
                task = what_if_background_analysis.delay(
                    user_email=request.user.email,
                    symbol=symbol,
                    investment_date_str=investment_date_str,
                    end_date_str=end_date_str,
                    investment_amount=investment_amount
                )
                task_id = task.id
                return redirect(f"/what-if/?task_id={task_id}")   # 🔥 Redirect immediately with task_id!

            if not task_id:
                # Instant historical analysis
                stock = yf.Ticker(symbol)
                data = stock.history(start=investment_date - datetime.timedelta(days=1), end=end_date + datetime.timedelta(days=1))
                data.index = data.index.date  # Force pure dates

                if data.empty:
                    raise ValueError(f"No data found for {symbol} between {investment_date} and {end_date}.")

                if investment_date in data.index:
                    investment_price = data.loc[investment_date]['Open']
                    if np.isnan(investment_price):
                        investment_price = data.loc[investment_date]['Close']
                else:
                    raise ValueError("No available data for investment date. Market may have been closed.")

                if end_date in data.index:
                    final_price = data.loc[end_date]['Close']
                else:
                    final_price = data.iloc[-1]['Close']

                final_value = (investment_amount / investment_price) * final_price
                percentage_change = ((final_value - investment_amount) / investment_amount) * 100

                graph_dates = [d.strftime("%Y-%m-%d") for d in data.index]
                graph_prices = [float(p) for p in data['Close'].tolist()]

                result = {
                    'symbol': symbol,
                    'investment_date': investment_date,
                    'end_date': end_date,
                    'initial_price': float(round(investment_price, 2)),
                    'final_price': float(round(final_price, 2)),
                    'investment_amount': float(round(investment_amount, 2)),
                    'final_value': float(round(final_value, 2)),
                    'percentage_change': float(round(percentage_change, 2)),
                    'graph_dates': graph_dates,
                    'graph_prices': graph_prices,
                }

        except Exception as e:
            error_message = f"Error: {str(e)}"

    user_past_tasks = WhatIfTaskResult.objects.filter(
        user_email=request.user.email,
        status='completed'
    ).order_by('-created_at')
    return render(request, 'what_if.html', {
        'result': result,
        'error_message': error_message,
        'symbol': request.GET.get('symbol', ''),
        'investment_date': request.GET.get('investment_date', ''),
        'end_date': request.GET.get('end_date', ''),
        'investment_amount': request.GET.get('investment_amount', ''),
        'today': datetime.date.today(),
        'task_id': task_id,
        'user_past_tasks': user_past_tasks,
    })

@require_GET
@login_required(login_url='/accounts/login/')
def what_if_status(request):
    task_id = request.GET.get("task_id")
    if not task_id:
        return JsonResponse({"status": "error", "message": "Missing task_id"}, status=400)

    try:
        task = WhatIfTaskResult.objects.get(task_id=task_id)
    except WhatIfTaskResult.DoesNotExist:
        return JsonResponse({"status": "pending"}, status=202)

    if task.status == "pending":
        return JsonResponse({"status": "pending"}, status=202)
    elif task.status == "completed":
        return JsonResponse({
            "status": "completed",
            "result_url": f"/what-if-result/?task_id={task_id}",
        }, status=200)
    elif task.status == "failed":
        return JsonResponse({
            "status": "failed",
            "message": task.error_message,
        }, status=500)
    else:
        return JsonResponse({"status": "error", "message": "Unknown status."}, status=500)

@login_required(login_url='/accounts/login/')
def what_if_result(request):
    task_id = request.GET.get('task_id')

    if not task_id:
        return redirect('what_if_analysis')

    try:
        task = WhatIfTaskResult.objects.get(task_id=task_id)
    except WhatIfTaskResult.DoesNotExist:
        return render(request, 'what_if.html', {
            'error_message': "Your analysis is not ready yet. Please check again later.",
            'today': datetime.date.today(),
        })

    if task.status == "completed":
        return render(request, 'what_if_result.html', {
            'result': task.result_json,
            'today': datetime.date.today(),
        })
    elif task.status == "failed":
        return render(request, 'what_if.html', {
            'error_message': task.error_message or "Something went wrong.",
            'today': datetime.date.today(),
        })
    else:
        return render(request, 'what_if.html', {
            'error_message': "Your analysis is still running. Please wait a few minutes.",
            'today': datetime.date.today(),
        })

@login_required
def get_quiz_questions(request):
    user_progress, _ = UserQuizProgress.objects.get_or_create(
        user=request.user, defaults={'level': QuizLevel.objects.get(name="beginner")}
    )

    level = user_progress.level
    quiz_number = int(request.GET.get("quiz", 1))  # Get quiz number from frontend

    # Get all questions for the current level
    all_questions = list(QuizQuestion.objects.filter(level=level).order_by("id"))
    quiz_size = max(len(all_questions) // 4, 1)  # Split questions among 4 quizzes

    # Determine the range of questions for this quiz
    start_index = (quiz_number - 1) * quiz_size
    end_index = min(start_index + quiz_size, len(all_questions))

    selected_questions = all_questions[start_index:end_index]

    if not selected_questions:
        return JsonResponse({"message": "No more questions available for this quiz."}, status=400)

    question_list = [{
        "id": q.id,
        "question": q.question_text,
        "options": {"A": q.option_a, "B": q.option_b, "C": q.option_c, "D": q.option_d}
    } for q in selected_questions]

    return JsonResponse({"questions": question_list, "level": level.name})

@login_required
def submit_quiz_answers(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)

    try:
        data = request.POST
        user_progress = UserQuizProgress.objects.get(user=request.user)
        quiz_number = request.GET.get("quiz", "1")  # Default to quiz 1

        correct_answers = sum(
            1 for q_id, answer in data.items() if QuizQuestion.objects.get(id=q_id).correct_answer == answer
        )

        earned_xp = correct_answers * 10

        # Update XP for this quiz
        user_progress.xp_per_quiz[quiz_number] = earned_xp
        user_progress.update_xp()
        user_progress.save()

        total_xp = user_progress.get_total_xp()

        return JsonResponse({
            "message": f"Total XP: {total_xp}",
            "xp_gained": earned_xp,
            "current_xp": total_xp,
        })

    except Exception as e:
        return JsonResponse({"error": "An error occurred."}, status=500)

@login_required
def get_user_progress(request):
    user_progress = get_object_or_404(UserQuizProgress, user=request.user)

    level_requirements = {
        "beginner": 120,
        "intermediate": 260,
        "advanced": 420,
        "expert": 600,
    }
    xp_needed = level_requirements.get(user_progress.level.name, 600)
    total_xp = user_progress.get_total_xp()
    quiz_xp = {}
    for quiz, xp in user_progress.xp_per_quiz.items():
        quiz_xp[int(quiz)] = xp

    return JsonResponse({
        "level": user_progress.level.name,
        "xp_points": total_xp,
        "xp_needed": xp_needed,
        "quiz_xp": quiz_xp,
        "quizzes_completed": len(user_progress.xp_per_quiz)
    })

def level_up(request):
    user_progress = get_object_or_404(UserQuizProgress, user=request.user)

    # Call the model's level_up function (which handles everything)
    user_progress.level_up()

    return JsonResponse({"message": "You have advanced to the next level!", "new_level": user_progress.level.name})

@login_required(login_url='/accounts/login/')
def quiz_page(request):
    return render(request, 'quiz.html')

@login_required
def get_quiz_history(request):
    user_progress = get_object_or_404(UserQuizProgress, user=request.user)

    levels = ["beginner", "intermediate", "advanced", "expert"]
    history = {}
    user_level_index = levels.index(user_progress.level.name)  # Get index of current level

    for level in levels[:user_level_index]:  # Only process completed levels
        history[level] = []
        all_questions = list(QuizQuestion.objects.filter(level__name=level).order_by("id"))

        # Ensure we process all four quizzes for this level
        for quiz_num in range(1, 5):
            quiz_size = max(len(all_questions) // 4, 1)  # Split questions evenly
            start_index = (quiz_num - 1) * quiz_size
            end_index = min(start_index + quiz_size, len(all_questions))

            selected_questions = all_questions[start_index:end_index]
            if not selected_questions:
                continue  # Skip empty quizzes

            quiz_data = {
                "quiz_number": quiz_num,
                "questions": []
            }

            for question in selected_questions:
                quiz_data["questions"].append({
                    "question": question.question_text,
                    "correct_answer": f"{question.correct_answer}: {getattr(question, f'option_{question.correct_answer.lower()}')}",  # Full correct answer
                    "explanation": question.explanation or "No explanation provided."
                })

            history[level].append(quiz_data)

    return JsonResponse({"history": history, "completed_levels": user_level_index > 0})

@login_required
def get_leaderboard(request):
    # Get top users sorted by XP first, then by the time they achieved it
    leaderboard = UserQuizProgress.objects.select_related('user').order_by('-total_xp', 'last_xp_update')[:10]

    leaderboard_data = [
        {
            "username": progress.user.username,
            "xp": progress.total_xp,
            "rank": index + 1
        }
        for index, progress in enumerate(leaderboard)
    ]

    return JsonResponse({"leaderboard": leaderboard_data})

@login_required(login_url='/accounts/login/')
def leaderboard_page(request):
    return render(request, 'leaderboard.html')

@require_GET
def api_stock_data(request):
    symbol = request.GET.get("symbol", "AAPL")
    period = request.GET.get("period", "1y")

    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)

        if data.empty:
            return JsonResponse({"error": "No data found"}, status=404)

        data.reset_index(inplace=True)
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data['Date'] = data['Date'].astype(str)

        return JsonResponse(data.to_dict(orient="records"), safe=False)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)