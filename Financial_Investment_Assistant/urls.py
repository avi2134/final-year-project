from django.contrib import admin
from django.urls import path, include
from Financial_Investment_Assistant import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('signup/', views.SignUpView.as_view(), name='signup'),
    path('accounts/', include("django.contrib.auth.urls")),
]