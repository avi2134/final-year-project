from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import now

class QuizLevel(models.Model):
    LEVEL_CHOICES = [
        ("beginner", "Beginner"),
        ("intermediate", "Intermediate"),
        ("advanced", "Advanced"),
        ("expert", "Expert"),
    ]
    name = models.CharField(max_length=20, choices=LEVEL_CHOICES, unique=True)
    passing_score = models.IntegerField(default=60)  # % required to pass

    def __str__(self):
        return self.name

class UserQuizProgress(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    level = models.ForeignKey(QuizLevel, on_delete=models.SET_NULL, null=True)
    xp_per_quiz = models.JSONField(default=dict)  # Tracks XP per quiz { "1": 0, "2": 0, "3": 0, "4": 0 }
    total_xp = models.IntegerField(default=0)  # Stores cumulative XP across all levels
    last_xp_update = models.DateTimeField(default=now)

    def get_total_xp(self):
        """Calculates total XP dynamically from all quizzes"""
        return sum(self.xp_per_quiz.values()) + self.total_xp

    def level_up(self):
        """Upgrade user to the next level and carry over XP."""
        level_order = ["beginner", "intermediate", "advanced", "expert"]
        level_requirements = {
            "beginner": 120,
            "intermediate": 260,
            "advanced": 420,
            "expert": 600,
        }

        if self.level:
            current_index = level_order.index(self.level.name)
            xp_needed = level_requirements[self.level.name]
            current_total_xp = self.get_total_xp()

            if current_total_xp >= xp_needed and current_index < len(level_order) - 1:
                next_level_name = level_order[current_index + 1]
                next_level = QuizLevel.objects.get(name=next_level_name)

                self.total_xp = current_total_xp
                self.xp_per_quiz = {}
                self.level = next_level
                self.save()

    def update_xp(self):
        self.last_xp_update = now()
        self.save(update_fields=['last_xp_update'])

    def __str__(self):
        return f"{self.user.username} - {self.level.name if self.level else 'No Level'}"

class QuizQuestion(models.Model):
    level = models.ForeignKey(QuizLevel, on_delete=models.CASCADE)  # Links to quiz levels
    question_text = models.TextField()
    option_a = models.CharField(max_length=255)
    option_b = models.CharField(max_length=255)
    option_c = models.CharField(max_length=255)
    option_d = models.CharField(max_length=255)
    correct_answer = models.CharField(max_length=1, choices=[("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")])
    explanation = models.CharField(blank=True, null=True, max_length=10000)  # Optional explanation for the answer

    def __str__(self):
        return f"{self.level.name} - {self.question_text[:50]}..."

class WatchedStock(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watchlist')
    symbol = models.CharField(max_length=10)
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.symbol}"

class WhatIfTaskResult(models.Model):
    task_id = models.CharField(max_length=255, unique=True)
    user_email = models.EmailField()
    status = models.CharField(max_length=20, choices=[('pending', 'Pending'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending')
    result_json = models.JSONField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)