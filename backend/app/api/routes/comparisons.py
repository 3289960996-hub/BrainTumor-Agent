"""Longitudinal quantitative MRI comparison endpoints."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import FileResponse

from backend.app.schemas.imaging import (
    ComparisonCreateRequest,
    ComparisonResponse,
    ComparisonTaskResponse,
)
from backend.app.services.dependencies import (
    get_comparison_task_repository,
    get_longitudinal_comparison_service,
)
from backend.app.services.errors import TaskQueueUnavailableError
from backend.app.tasks.celery_app import celery_app
from longitudinal.service import (
    LongitudinalComparisonService,
    comparison_id_for,
)
from longitudinal.storage import ComparisonTaskRepository

router = APIRouter()


def _worker_available() -> bool:
    try:
        replies = celery_app.control.inspect(timeout=2).ping()
    except Exception:
        return False
    return bool(replies)


def _task_response(record: dict, request: Request) -> ComparisonTaskResponse:
    payload = dict(record)
    payload.pop("request", None)
    payload["result_url"] = (
        str(
            request.url_for(
                "get_comparison",
                comparison_id=record["comparison_id"],
            )
        )
        if record["status"] == "succeeded"
        else None
    )
    return ComparisonTaskResponse.model_validate(payload)


@router.post(
    "/comparison-tasks",
    response_model=ComparisonTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="提交异步随访空间对比任务",
)
def create_comparison_task(
    payload: ComparisonCreateRequest,
    request: Request,
    service: Annotated[
        LongitudinalComparisonService,
        Depends(get_longitudinal_comparison_service),
    ],
    tasks: Annotated[
        ComparisonTaskRepository,
        Depends(get_comparison_task_repository),
    ],
) -> ComparisonTaskResponse:
    service.validate_request(
        baseline_case_id=payload.baseline_case_id,
        followup_case_id=payload.followup_case_id,
        baseline_study_date=payload.baseline_study_date,
        followup_study_date=payload.followup_study_date,
    )
    comparison_id = comparison_id_for(
        payload.patient_group_id,
        payload.baseline_case_id,
        payload.followup_case_id,
        payload.baseline_study_date,
        payload.followup_study_date,
    )
    request_payload = {
        "patient_group_id": payload.patient_group_id,
        "baseline_case_id": payload.baseline_case_id,
        "followup_case_id": payload.followup_case_id,
        "baseline_study_date": payload.baseline_study_date.isoformat(),
        "followup_study_date": payload.followup_study_date.isoformat(),
    }
    task, created = tasks.create(
        comparison_id=comparison_id,
        request_payload=request_payload,
    )
    if created:
        if not _worker_available():
            tasks.update(
                task["task_id"],
                status="failed",
                stage="dispatch_failed",
                message="空间对比Worker未就绪",
                error_code="task_queue_unavailable",
                error_message="Redis或Celery Worker当前不可用",
            )
            raise TaskQueueUnavailableError()
        try:
            celery_app.send_task(
                "comparison.run",
                args=[task["task_id"]],
                task_id=task["task_id"],
            )
        except Exception as exc:
            tasks.update(
                task["task_id"],
                status="failed",
                stage="dispatch_failed",
                message="空间对比任务无法投递",
                error_code="task_queue_unavailable",
                error_message="Redis或Celery Worker当前不可用",
            )
            raise TaskQueueUnavailableError() from exc
    return _task_response(task, request)


@router.get(
    "/comparison-tasks/{task_id}",
    response_model=ComparisonTaskResponse,
    summary="查询空间对比任务状态和进度",
)
def get_comparison_task(
    task_id: str,
    request: Request,
    tasks: Annotated[
        ComparisonTaskRepository,
        Depends(get_comparison_task_repository),
    ],
) -> ComparisonTaskResponse:
    return _task_response(tasks.get(task_id), request)


@router.post(
    "/comparison-tasks/{task_id}/cancel",
    response_model=ComparisonTaskResponse,
    summary="请求取消空间对比任务",
)
def cancel_comparison_task(
    task_id: str,
    request: Request,
    tasks: Annotated[
        ComparisonTaskRepository,
        Depends(get_comparison_task_repository),
    ],
) -> ComparisonTaskResponse:
    task = tasks.get(task_id)
    if task["status"] == "queued":
        task = tasks.update(
            task_id,
            status="cancelled",
            stage="cancelled",
            message="排队中的空间对比任务已取消",
            finished_at=datetime.now(UTC).isoformat(),
        )
        try:
            celery_app.control.revoke(task_id)
        except Exception:
            pass
    elif task["status"] == "running":
        task = tasks.update(
            task_id,
            status="cancel_requested",
            message="已请求取消，将在当前安全阶段结束后停止",
        )
    return _task_response(task, request)


@router.post(
    "/comparisons",
    response_model=ComparisonResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建两个已分析MRI病例的定量随访对比",
)
def create_comparison(
    payload: ComparisonCreateRequest,
    request: Request,
    service: Annotated[
        LongitudinalComparisonService,
        Depends(get_longitudinal_comparison_service),
    ],
) -> ComparisonResponse:
    result = service.create(
        patient_group_id=payload.patient_group_id,
        baseline_case_id=payload.baseline_case_id,
        followup_case_id=payload.followup_case_id,
        baseline_study_date=payload.baseline_study_date,
        followup_study_date=payload.followup_study_date,
    )
    return ComparisonResponse.model_validate(_with_artifact_urls(result, request))


@router.get(
    "/comparisons/{comparison_id}",
    response_model=ComparisonResponse,
    summary="读取已保存的定量随访对比",
)
def get_comparison(
    comparison_id: str,
    request: Request,
    service: Annotated[
        LongitudinalComparisonService,
        Depends(get_longitudinal_comparison_service),
    ],
) -> ComparisonResponse:
    return ComparisonResponse.model_validate(
        _with_artifact_urls(service.get(comparison_id), request)
    )


@router.get(
    "/comparisons/{comparison_id}/artifacts/{artifact_key}",
    response_class=FileResponse,
    name="download_comparison_artifact",
    summary="下载配准影像或空间变化Mask",
)
def download_comparison_artifact(
    comparison_id: str,
    artifact_key: str,
    service: Annotated[
        LongitudinalComparisonService,
        Depends(get_longitudinal_comparison_service),
    ],
) -> FileResponse:
    target = service.comparisons.artifact(comparison_id, artifact_key)
    media_type = "application/gzip" if target.name.endswith(".nii.gz") else "text/plain"
    return FileResponse(path=target, media_type=media_type, filename=target.name)


def _with_artifact_urls(result: dict, request: Request) -> dict:
    payload = dict(result)
    spatial = payload.get("spatial_comparison")
    if not isinstance(spatial, dict):
        return payload
    spatial_payload = dict(spatial)
    artifacts = spatial_payload.get("artifacts", {})
    if isinstance(artifacts, dict):
        spatial_payload["artifacts"] = {
            key: str(
                request.url_for(
                    "download_comparison_artifact",
                    comparison_id=payload["comparison_id"],
                    artifact_key=key,
                )
            )
            for key in artifacts
        }
    spatial_payload["baseline_t1ce_url"] = str(
        request.url_for(
            "download_modality",
            case_id=payload["baseline_case_id"],
            modality="t1ce",
        )
    )
    payload["spatial_comparison"] = spatial_payload
    return payload
