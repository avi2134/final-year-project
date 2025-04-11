from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import UserQuizProgress, QuizLevel

@receiver(post_save, sender=User)
def create_user_quiz_progress(sender, instance, created, **kwargs):
    if created:
        beginner_level = QuizLevel.objects.get(name="beginner")
        UserQuizProgress.objects.create(user=instance, level=beginner_level)