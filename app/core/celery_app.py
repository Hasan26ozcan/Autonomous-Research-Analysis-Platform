"""
app/core/celery_app.py
======================
Celery application instance — shared between API (for sending tasks)
and worker (for executing tasks).
"""
from celery import Celery
from app.core.config import settings

# Create the Celery application instance
celery_app = Celery(
    "arap",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Update configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Disable the deprecation warning about broker_connection_retry (as seen in logs)
    broker_connection_retry_on_startup=True,
    # Ensure the worker attempts to connect on startup
    broker_connection_retry=True,
    # Task result expiration time (1 day)
    result_expires=86400,
)

# Auto-discover tasks from the app.services module (finds tasks.py)
celery_app.autodiscover_tasks(["app.services"])