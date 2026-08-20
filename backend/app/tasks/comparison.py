"""Persistent Celery task for longitudinal spatial comparison."""

from datetime import UTC, date, datetime
from typing import Any

from celery import Task

from backend.app.core.config import get_settings
from backend.app.services.dependencies import (
    get_comparison_task_repository,
    get_longitudinal_comparison_service,
)
from backend.app.services.errors import BackendServiceError
from backend.app.tasks.celery_app import celery_app
from longitudinal.service import ComparisonCancellationRequested


def _now() -> str:
    return datetime.now(UTC).isoformat()


@celery_app.task(bind=True, name="comparison.run", acks_late=True)
def run_comparison(self: Task, task_id: str) -> dict[str, Any]:
    settings = get_settings()
    tasks = get_comparison_task_repository()
    record = tasks.get(task_id)
    if record["status"] in {"cancelled", "failed", "succeeded"}:
        return record
    if record["status"] == "cancel_requested":
        return tasks.update(
            task_id,
            status="cancelled",
            stage="cancelled",
            message="空间对比在开始前已取消",
            finished_at=_now(),
        )
    tasks.update(
        task_id,
        status="running",
        stage="starting",
        progress=2,
        message="Worker已开始执行空间对比",
        started_at=record.get("started_at") or _now(),
        attempt=int(record.get("attempt", 0)) + 1,
        error_code=None,
        error_message=None,
    )

    def progress(stage: str, percent: int, message: str) -> None:
        current = tasks.get(task_id)
        if current["status"] == "cancel_requested":
            raise ComparisonCancellationRequested
        tasks.update(task_id, stage=stage, progress=percent, message=message)

    request = record["request"]
    try:
        result = get_longitudinal_comparison_service().create(
            patient_group_id=request["patient_group_id"],
            baseline_case_id=request["baseline_case_id"],
            followup_case_id=request["followup_case_id"],
            baseline_study_date=date.fromisoformat(request["baseline_study_date"]),
            followup_study_date=date.fromisoformat(request["followup_study_date"]),
            progress=progress,
        )
        current = tasks.get(task_id)
        if current["status"] == "cancel_requested":
            return tasks.update(
                task_id,
                status="cancelled",
                stage="cancelled",
                message="空间计算已结束，但任务按取消请求停止",
                finished_at=_now(),
            )
        return tasks.update(
            task_id,
            status="succeeded",
            stage="completed",
            progress=100,
            message="随访定量与空间对比已完成",
            comparison_id=result["comparison_id"],
            finished_at=_now(),
        )
    except ComparisonCancellationRequested:
        return tasks.update(
            task_id,
            status="cancelled",
            stage="cancelled",
            message="空间对比已在安全阶段边界停止",
            finished_at=_now(),
        )
    except Exception as exc:
        error_code = exc.code if isinstance(exc, BackendServiceError) else "comparison_failed"
        public_message = (
            exc.message
            if isinstance(exc, BackendServiceError)
            else "空间对比执行失败，请检查影像几何和Worker日志"
        )
        retries = int(getattr(self.request, "retries", 0))
        if retries < settings.analysis_task_max_retries:
            tasks.update(
                task_id,
                status="queued",
                stage="retrying",
                message=f"空间对比失败，将自动重试：{public_message}",
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
            message="空间对比失败",
            error_code=error_code,
            error_message=public_message,
            finished_at=_now(),
        )
        raise
