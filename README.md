
# Financial Investment Assistant — InvestGrow

A final-year Django project that provides an educational platform to explore stock investment strategies using historical data and predictive modeling. Users can analyze hypothetical investments, receive future predictions via email, and track their progress interactively through quizzes and dashboards.

---

## Project Structure

```
Third_Year_Project/
├── manage.py
├── db.sqlite3 / PostgreSQL
├── cached_models/                     # Trained LSTM models stored here
├── static/
│   ├── js/                            # All JavaScript files
│   └── style/                         # CSS stylesheets
├── templates/
│   ├── account/                       # User auth templates
│   ├── partials/                      # Partial components
│   └── *.html                         # Page templates
├── Financial_Investment_Assistant/
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── signals.py
│   ├── tasks.py                       # Celery background task for predictions
│   ├── urls.py
│   ├── views.py
│   └── migrations/
└── Third_Year_Project/
    ├── __init__.py
    ├── celery.py                      # Celery app setup
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

---

## System Requirements

- **OS**: Windows 10/11, macOS, or any Linux distro
- **Python**: 3.9+
- **PostgreSQL** (running locally on port 5432)
- **Redis** (running on port 6380)
- **Pip** (Python package manager)
- **Internet connection** (for stock data & email API)

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd as1473
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Required Python Libraries

```bash
pip install django celery redis yfinance numpy pandas tensorflow joblib django-allauth django-extensions scikit-learn
```

---

## Configuration

### Database Setup (PostgreSQL)

```bash
python manage.py makemigrations
python manage.py migrate
```

### Create Admin Superuser

```bash
python manage.py createsuperuser
```

---

## Running the Project

### 1. Start Redis on Port 6380

```bash
redis-server --port 6380
```

### 2. Start the Celery Worker

```bash
celery -A Third_Year_Project worker --loglevel=info --pool=solo
```

### 3. Start Django Development Server

```bash
python manage.py runserver
```

Then access the project at: [http://127.0.0.1:8000](http://127.0.0.1:8000)


## Features

- What-If Investment Simulator
- LSTM-based future stock price prediction
- Background task system with Celery and Redis
- Email delivery when results are ready
- Financial knowledge quizzes
- Leaderboards and progress tracking
- Account management via Django-AllAuth


## Security Notes

- Set `DEBUG = False` in production
- Use `.env` file to hide sensitive settings
- Limit `ALLOWED_HOSTS`

---

## Author

**as1473** — 2025 Final Year Project (InvestGrow)