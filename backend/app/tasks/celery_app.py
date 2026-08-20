"""Celery application for resource-intensive MRI analysis."""

from celery import Celery

from backend.app.core.config import get_settings

settings = get_settings()
celery_app = Celery(
    "brain_tumor_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["backend.app.tasks.analysis", "backend.app.tasks.comparison"],
)
celery_app.conf.update(
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    timezone="UTC",
)
