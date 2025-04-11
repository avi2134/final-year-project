from django.contrib import admin
from .models import *

admin.site.register(QuizLevel)
admin.site.register(UserQuizProgress)
admin.site.register(QuizQuestion)