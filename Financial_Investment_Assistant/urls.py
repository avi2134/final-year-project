from django.contrib import admin
from django.template.context_processors import request
from django.urls import path, include
from Financial_Investment_Assistant import views
from Financial_Investment_Assistant.views import signup_view
from django.contrib.auth import views as auth_views

urlpatterns = [
    # path('admin/', admin.site.urls),
    # path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('', views.index, name='index'),
    path('signup/', signup_view, name='signup'),
    path('accounts/', include('allauth.urls')),
    path('landing/', views.landing_page, name='landing'),
    path("api/stats/", views.stats_api, name="api-stats"),
    path("help/", views.help_page, name="help_page"),
    path("profile/", views.profile_view, name="profile"),
    path('buy_sell/', views.buy_sell, name='buy_sell'),
    path('what-if/', views.what_if_analysis, name='what_if_analysis'),
    path('fetch-news/', views.fetch_news, name='fetch_news'),
    path('fetch-trending-stocks/', views.fetch_trending_stocks, name='fetch_trending_stocks'),
    path("fetch-stock-search/", views.fetch_stock_search, name="fetch_stock_search"),
    path('fetch-gainers-losers/', views.fetch_gainers_losers, name="fetch_gainers_losers"),
    path("fetch-market-summary/", views.fetch_market_summary, name="fetch_market_summary"),
    path("add-to-watchlist/", views.add_to_watchlist, name="add_to_watchlist"),
    path("delete-stock/", views.delete_stock, name="delete_stock"),
    path("fetch-market-chart/", views.fetch_market_chart_data, name="fetch_market_chart"),
    path('quiz/', views.quiz_page, name='quiz_page'),  # Frontend quiz page
    path('api/get-quiz-questions/', views.get_quiz_questions, name='get_quiz_questions'),
    path('api/submit-quiz-answers/', views.submit_quiz_answers, name='submit_quiz_answers'),
    path('api/get-user-progress/', views.get_user_progress, name='get_user_progress'),
    path('api/level-up/', views.level_up, name='level_up'),
    path('api/get-quiz-history/', views.get_quiz_history, name='get_quiz_history'),
    path('leaderboard/', views.leaderboard_page, name='leaderboard_page'),
    path('api/get-leaderboard/', views.get_leaderboard, name='get_leaderboard'),
    path("api/stock_data/", views.api_stock_data, name="api_stock_data"),
]