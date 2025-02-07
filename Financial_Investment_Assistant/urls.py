from django.contrib import admin
from django.template.context_processors import request
from django.urls import path, include
from Financial_Investment_Assistant import views
from Financial_Investment_Assistant.views import signup_view
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('', views.index, name='index'),
    path('signup/', signup_view, name='signup'),
    path('accounts/', include("django.contrib.auth.urls")),
    path('buy_sell/', views.buy_sell, name='buy_sell'),
    path('what-if/', views.what_if_analysis, name='what_if_analysis'),
    path('fetch-news/', views.fetch_news, name='fetch_news'),
    path('fetch-trending-stocks/', views.fetch_trending_stocks, name='fetch_trending_stocks'),
    path("fetch-stock-search/", views.fetch_stock_search, name="fetch_stock_search"),

]