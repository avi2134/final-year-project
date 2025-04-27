import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Third_Year_Project.settings')

app = Celery('Third_Year_Project')

app.config_from_object('django.conf:settings', namespace='CELERY')

# Automatically discover tasks from all apps
app.autodiscover_tasks()