from celery import shared_task
from app.services.ingest_service import run_ingest_pipeline
from app.core.logging import logger

@shared_task(bind=True, max_retries=2, soft_time_limit=1800, time_limit=1860)
def ingest_document_task(self, file_content: bytes, filename: str, user_id: str = "default"):
    """
    Celery worker'da çalışan asıl ingest task'i.
    """
    try:
        result = run_ingest_pipeline(file_content, filename, user_id)
        return result
    except Exception as exc:
        logger.warning(f"Task failed, retrying... Error: {exc}")
        # 60 saniye sonra tekrar dene
        self.retry(exc=exc, countdown=60)