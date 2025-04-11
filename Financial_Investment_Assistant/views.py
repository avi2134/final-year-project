from lib2to3.fixes.fix_input import context
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.urls import reverse_lazy
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import CreateView
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.urls import reverse
from verify_email.email_handler import send_verification_email
from .forms import CustomUserCreationForm
import requests
import plotly.graph_objs as go
import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import datetime
from django.shortcuts import get_object_or_404
from .models import *


def fetch_stock_search(request):
    query = request.GET.get("query", "").strip()
    if not query:
        return JsonResponse({"error": "No search query provided"}, status=400)

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

        return JsonResponse({"bestMatches": results})

    except Exception as e:
        return JsonResponse({"error": f"Failed to fetch data: {str(e)}"}, status=500)

def fetch_stock_data(symbol, time_range):
    api_key = 'E4AV7OCP8Y7L5D36'
    url = "https://www.alphavantage.co/query"
    time_function = {
        'daily': 'TIME_SERIES_DAILY',
        'weekly': 'TIME_SERIES_WEEKLY',
        'monthly': 'TIME_SERIES_MONTHLY'
    }.get(time_range, 'TIME_SERIES_DAILY')  # Default to daily

    params = {
        'function': time_function,
        'symbol': symbol,
        'apikey': api_key,
    }
    response = requests.get(url, params=params)

    data = response.json()

    if time_range == 'daily':
        return data.get('Time Series (Daily)', {})
    elif time_range == 'weekly':
        return data.get('Weekly Time Series', {})
    elif time_range == 'monthly':
        return data.get('Monthly Time Series', {})
    return {}
@login_required(login_url='/accounts/login/')
def index(request):
    stock_symbol = request.GET.get('symbol', 'AAPL')
    time_range = request.GET.get('time_range', 'daily').capitalize()

    stock_data = fetch_stock_data(stock_symbol, time_range.lower())
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
        'time_range': time_range,  # Already capitalised
        'error_message': error_message
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

# Fetch Trending Stocks from Yahoo Finance
def fetch_trending_stocks(request):
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

# Buy/Sell page view
@login_required(login_url='/accounts/login/')
def buy_sell(request):
    chart_image = None
    error_message = None
    recommendation = None

    if request.method == 'GET' and 'symbol' in request.GET:
        symbol = request.GET.get('symbol')
        shares_owned = request.GET.get('shares_owned', '').strip()
        amount_invested = request.GET.get('amount_invested', '').strip()
        risk_level = request.GET.get('risk_level', 'moderate').strip()  # Default: moderate

        # Define the period to consider for analysis (e.g., 6 months ≈ 180 trading days)
        recent_days = 180  # Use 6 months of data by default

        try:
            # Fetch historical data
            stock = yf.Ticker(symbol)
            data = stock.history(period="6mo")
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

            # Use only the recent data for analysis
            recent_data = data.tail(recent_days)

            # Calculate indicators on the recent data
            recent_data['EMA_5'] = calculate_ema(recent_data, 5)
            recent_data['EMA_8'] = calculate_ema(recent_data, 8)
            recent_data['EMA_13'] = calculate_ema(recent_data, 13)
            recent_data['RSI'] = calculate_rsi(recent_data)

            print(recent_data)

            # Generate signals with risk adjustment
            recent_data = generate_signals(recent_data, risk_level)

            # Get the latest price and recommendation
            latest_price = recent_data['Close'].iloc[-1]
            latest_signal = recent_data['Signal'].iloc[-1]

            if latest_signal == 1:
                recommendation = f"Consider buying {symbol}. Risk Level: {risk_level.capitalize()}."
            elif latest_signal == -1:
                recommendation = f"Consider selling your holdings of {symbol}. Risk Level: {risk_level.capitalize()}."
            else:
                recommendation = f"No strong Buy/Sell signal detected for {symbol}. Risk Level: {risk_level.capitalize()}."

            # Plot chart using the recent data
            chart_image = plot_candlestick(recent_data)

        except Exception as e:
            error_message = f"An error occurred: {e}"

    return render(request, 'buy_sell.html', {
        'chart_image': chart_image,
        'error_message': error_message,
        'recommendation': recommendation,
        'symbol': symbol if 'symbol' in request.GET else '',
        'shares_owned': shares_owned if 'shares_owned' in request.GET else '',
        'amount_invested': amount_invested if 'amount_invested' in request.GET else '',
        'risk_level': risk_level if 'risk_level' in request.GET else 'moderate',
    })
@login_required(login_url='/accounts/login/')
def what_if_analysis(request):
    result = None
    chart_image = None
    error_message = None

    if request.method == 'GET' and 'symbol' in request.GET:
        symbol = request.GET.get('symbol').strip().upper()
        investment_date = request.GET.get('investment_date')
        investment_amount = request.GET.get('investment_amount')

        try:
            investment_date = datetime.datetime.strptime(investment_date, "%Y-%m-%d").date()
            today = datetime.date.today()

            if investment_date >= today:
                error_message = "Investment date must be in the past."
            else:
                # Fetch historical data
                stock = yf.Ticker(symbol)
                data = stock.history(start=investment_date, end=today)

                if data.empty:
                    error_message = f"No data found for {symbol} on {investment_date}."
                else:
                    # Get price on investment date
                    initial_price = data.iloc[0]['Close']
                    # Get price today
                    final_price = data.iloc[-1]['Close']
                    # Calculate final investment value
                    final_value = (float(investment_amount) / initial_price) * final_price
                    percentage_change = ((final_value - float(investment_amount)) / float(investment_amount)) * 100

                    result = {
                        'symbol': symbol,
                        'investment_date': investment_date,
                        'initial_price': round(initial_price, 2),
                        'final_price': round(final_price, 2),
                        'investment_amount': round(float(investment_amount), 2),
                        'final_value': round(final_value, 2),
                        'percentage_change': round(percentage_change, 2),
                    }

                    # Generate graph of stock price over time
                    plt.figure(figsize=(10, 5))
                    plt.plot(data.index, data['Close'], label=f"{symbol} Stock Price", color='blue')
                    plt.scatter(data.index[0], initial_price, color='green', label=f'Investment Date (£{initial_price})', zorder=3)
                    plt.scatter(data.index[-1], final_price, color='red', label=f'Today (£{final_price})', zorder=3)
                    plt.xlabel("Date")
                    plt.ylabel("Stock Price (£)")
                    plt.title(f"{symbol} Stock Growth Since {investment_date}")
                    plt.legend()
                    plt.grid()

                    # Convert graph to base64 for embedding in HTML
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight')
                    buffer.seek(0)
                    chart_image = base64.b64encode(buffer.read()).decode('utf-8')
                    buffer.close()

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render(request, 'what_if.html', {
        'result': result,
        'chart_image': chart_image,
        'error_message': error_message,
        'symbol': request.GET.get('symbol', ''),
        'investment_date': request.GET.get('investment_date', ''),
        'investment_amount': request.GET.get('investment_amount', ''),
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