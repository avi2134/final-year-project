# Generated by Django 5.1.2 on 2025-03-10 23:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Financial_Investment_Assistant', '0007_remove_quizquestion_explanation_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='quizquestion',
            name='explanation',
            field=models.TextField(blank=True, null=True),
        ),
    ]
