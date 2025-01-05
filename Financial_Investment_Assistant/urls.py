from django.contrib import admin
from django.template.context_processors import request
from django.urls import path, include
from Financial_Investment_Assistant import views
from Financial_Investment_Assistant.views import signup_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('signup/', signup_view, name='signup'),
    path('accounts/', include("django.contrib.auth.urls")),
]