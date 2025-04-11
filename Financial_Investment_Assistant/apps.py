from django.apps import AppConfig

class FinancialInvestmentAssistantConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Financial_Investment_Assistant'

    def ready(self):
        import Financial_Investment_Assistant.signals