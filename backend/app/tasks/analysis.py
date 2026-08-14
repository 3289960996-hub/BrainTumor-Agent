"""Persistent Celery MRI analysis task."""

from datetime import UTC, datetime
from typing import Any

from celery import Task

from backend.app.core.config import get_settings
from backend.app.services.analysis import AnalysisCancellationRequested
from backend.app.services.dependencies import (
    get_analysis_pipeline,
    get_analysis_task_repository,
)
from backend.app.services.errors import BackendServiceError
from backend.app.tasks.celery_app import celery_app


def _now() -> str:
    return datetime.now(UTC).isoformat()


@celery_app.task(bind=True, name="analysis.run", acks_late=True)
def run_analysis(self: Task, task_id: str, case_id: str) -> dict[str, Any]:
    settings = get_settings()
    tasks = get_analysis_task_repository()
    record = tasks.get(task_id)
    if record["status"] == "cancel_requested":
        return tasks.update(
            task_id,
            status="cancelled",
            stage="cancelled",
            message="任务在开始前已取消",
            finished_at=_now(),
        )
    tasks.update(
        task_id,
        status="running",
        stage="starting",
        progress=2,
        message="Worker已开始执行分析",
        started_at=record.get("started_at") or _now(),
        attempt=int(record.get("attempt", 0)) + 1,
        error_code=None,
        error_message=None,
    )

    def progress(stage: str, percent: int, message: str) -> None:
        current = tasks.get(task_id)
        if current["status"] == "cancel_requested":
            raise AnalysisCancellationRequested
        tasks.update(task_id, stage=stage, progress=percent, message=message)

    try:
        result = get_analysis_pipeline().analyze(case_id, progress=progress)
        current = tasks.get(task_id)
        if current["status"] == "cancel_requested":
            return tasks.update(
                task_id,
                status="cancelled",
                stage="cancelled",
                progress=100,
                message="分析已完成，但任务按取消请求停止后续处理",
                finished_at=_now(),
            )
        return tasks.update(
            task_id,
            status="succeeded",
            stage="completed",
            progress=100,
            message="MRI分析已完成",
            finished_at=_now(),
            mask_filename=result.mask_path.name,
        )
    except AnalysisCancellationRequested:
        get_analysis_pipeline().repository.write_status(case_id, "uploaded")
        return tasks.update(
            task_id,
            status="cancelled",
            stage="cancelled",
            message="分析已在安全阶段边界停止",
            finished_at=_now(),
        )
    except Exception as exc:
        error_code = exc.code if isinstance(exc, BackendServiceError) else "analysis_failed"
        public_message = (
            exc.message
            if isinstance(exc, BackendServiceError)
            else "MRI分析执行失败，请检查Worker日志和模型配置"
        )
        retries = int(getattr(self.request, "retries", 0))
        if retries < settings.analysis_task_max_retries:
            tasks.update(
                task_id,
                status="queued",
                stage="retrying",
                message=f"分析失败，将自动重试：{public_message}",
                error_code=error_code,
                error_message=public_message,
            )
            raise self.retry(
                exc=exc,
                countdown=settings.analysis_task_retry_delay_seconds,
                max_retries=settings.analysis_task_max_retries,
            ) from exc
        tasks.update(
            task_id,
            status="failed",
            stage="failed",
            message="MRI分析失败",
            error_code=error_code,
            error_message=public_message,
            finished_at=_now(),
        )
        raise
