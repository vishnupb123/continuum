from celery import Celery

from app.core.config import settings


celery_app = Celery(
    "continuum",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.journal_tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

celery_app.autodiscover_tasks(["app.tasks"])