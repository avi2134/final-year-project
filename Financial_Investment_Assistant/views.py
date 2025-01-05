from lib2to3.fixes.fix_input import context
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.urls import reverse
from verify_email.email_handler import send_verification_email
from .forms import CustomUserCreationForm
import requests
import plotly.graph_objs as go

def fetch_stock_data(symbol, time_range):
    API_KEY = 'E4AV7OCP8Y7L5D36'
    url = "https://www.alphavantage.co/query"
    time_function = {
        'daily': 'TIME_SERIES_DAILY',
        'weekly': 'TIME_SERIES_WEEKLY',
        'monthly': 'TIME_SERIES_MONTHLY'
    }.get(time_range, 'TIME_SERIES_DAILY')  # Default to daily

    params = {
        'function': time_function,
        'symbol': symbol,
        'apikey': API_KEY,
    }
    response = requests.get(url, params=params)
    print("Response JSON:", response.json())
    data = response.json()

    if time_range == 'daily':
        return data.get('Time Series (Daily)', {})
    elif time_range == 'weekly':
        return data.get('Weekly Time Series', {})
    elif time_range == 'monthly':
        return data.get('Monthly Time Series', {})
    return {}

def index(request):
    stock_symbol = request.GET.get('symbol', 'AAPL')
    time_range = request.GET.get('time_range', 'daily').capitalize()  # Capitalize here

    stock_data = fetch_stock_data(stock_symbol, time_range.lower())  # Use lowercase for API
    chart_html = None
    error_message = None

    if stock_data:
        # Prepare the data
        dates = list(stock_data.keys())[:30]
        close_prices = [float(stock_data[date]['4. close']) for date in dates]

        # Create a Plotly chart
        trace = go.Scatter(
            x=dates[::-1],
            y=close_prices[::-1],
            mode='lines+markers',
            name=f'{stock_symbol} Closing Prices',
            line=dict(color='blue')
        )
        layout = go.Layout(
            title=f"{stock_symbol} - {time_range} Closing Prices",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price (USD)"),
            template="plotly_dark",
            showlegend=True  # Add legend
        )
        fig = go.Figure(data=[trace], layout=layout)
        chart_html = fig.to_html(full_html=False)
    else:
        error_message = f"No data available for '{stock_symbol}'. Please check the symbol or time range."

    return render(request, 'index.html', {
        'chart_html': chart_html,
        'stock_symbol': stock_symbol,
        'time_range': time_range,  # Already capitalized
        'error_message': error_message
    })

def signup_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = True  # Skip email verification for now
            user.save()
            return redirect('login')  # Redirect to login or home page
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

