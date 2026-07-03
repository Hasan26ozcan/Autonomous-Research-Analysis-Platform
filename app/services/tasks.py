from __future__ import annotations

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "arap",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(task_track_started=True, task_time_limit=3600)


@celery_app.task(name="app.services.tasks.ping")
def ping() -> str:
    return "pong"
