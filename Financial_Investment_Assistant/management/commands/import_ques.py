import csv
from django.core.management.base import BaseCommand
from Financial_Investment_Assistant.models import QuizQuestion, QuizLevel

class Command(BaseCommand):
    help = "Import quiz questions from a CSV file"

    def handle(self, *args, **kwargs):
        file_path = "static/questions.csv"  # Ensure this file is in the correct location

        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            total_inserted = 0
            total_skipped = 0

            for row in reader:
                level, created = QuizLevel.objects.get_or_create(name=row["Level"].strip().lower())
                # ✅ Check if the question already exists to prevent duplicates
                existing_question = QuizQuestion.objects.filter(
                    level=level,
                    question_text=row["Question"].strip()
                ).first()

                if existing_question:
                    self.stdout.write(self.style.WARNING(f"⚠️ Skipping duplicate: {row['Question']}"))
                    total_skipped += 1
                    continue  # Skip inserting duplicate question

                QuizQuestion.objects.create(
                    level=level,
                    question_text=row["Question"].strip(),
                    option_a=row["Option_A"].strip(),
                    option_b=row["Option_B"].strip(),
                    option_c=row["Option_C"].strip(),
                    option_d=row["Option_D"].strip(),
                    correct_answer=row["Correct_Answer"].strip()[0].upper(),  # Ensure only "A", "B", "C", or "D"
                    explanation=row["Explanation"].strip(),
                )

                total_inserted += 1

        self.stdout.write(self.style.SUCCESS(f"✅ Successfully imported {total_inserted} new quiz questions!"))
        self.stdout.write(self.style.WARNING(f"⚠️ Skipped {total_skipped} duplicate questions."))
