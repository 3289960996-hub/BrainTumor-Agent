"""MRI上传、分析、报告和Agent问答接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import FileResponse

from backend.app.schemas.imaging import (
    AnalysisTaskResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    CaseRestoreResponse,
    ChatRequest,
    ChatResponse,
    MaskArtifact,
    ReportApplyRequest,
    ReportApplyResponse,
    ReportEditRequest,
    ReportEditResponse,
    ReportRequest,
    ReportResponse,
    TumorMetrics,
    UploadResponse,
)
from backend.app.services.chat import MedicalAgentChatService
from backend.app.services.dependencies import (
    get_analysis_task_repository,
    get_case_repository,
    get_chat_service,
    get_report_editing_service,
    get_report_service,
    get_upload_service,
)
from backend.app.services.errors import TaskQueueUnavailableError
from backend.app.services.reporting import (
    MedicalReportEditingService,
    MedicalReportService,
)
from backend.app.services.storage import AnalysisTaskRepository, CaseRepository
from backend.app.services.upload import MRIUploadService
from backend.app.tasks.analysis import run_analysis
from data_process.constants import MRIModality

router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
    summary="上传BraTS四模态MRI",
)
async def upload_mri(
    service: Annotated[MRIUploadService, Depends(get_upload_service)],
    t1: Annotated[UploadFile, File(description="T1 NIfTI")],
    t1ce: Annotated[UploadFile, File(description="T1ce NIfTI")],
    t2: Annotated[UploadFile, File(description="T2 NIfTI")],
    flair: Annotated[UploadFile, File(description="FLAIR NIfTI")],
    case_id: Annotated[
        str | None,
        Form(
            description="可选去标识化病例编号；不应使用姓名、住院号等患者标识"
        ),
    ] = None,
) -> UploadResponse:
    uploaded = await service.upload_case(
        {
            MRIModality.T1: t1,
            MRIModality.T1CE: t1ce,
            MRIModality.T2: t2,
            MRIModality.FLAIR: flair,
        },
        case_id=case_id,
    )
    return UploadResponse(
        case_id=uploaded.case_id,
        status="uploaded",
        modalities=uploaded.modality_files,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="执行MRI预处理、nnU-Net推理和指标计算",
)
def analyze_mri(
    payload: AnalyzeRequest,
    repository: Annotated[CaseRepository, Depends(get_case_repository)],
    tasks: Annotated[
        AnalysisTaskRepository,
        Depends(get_analysis_task_repository),
    ],
) -> AnalyzeResponse:
    repository.require_case(payload.case_id)
    task, created = tasks.create(payload.case_id)
    if created:
        try:
            run_analysis.apply_async(args=[task["task_id"], payload.case_id])
        except Exception as exc:
            tasks.update(
                task["task_id"],
                status="failed",
                stage="dispatch_failed",
                message="任务无法投递到分析队列",
                error_code="task_queue_unavailable",
                error_message="Redis或Celery不可用",
            )
            raise TaskQueueUnavailableError() from exc
    return AnalyzeResponse(
        case_id=payload.case_id,
        task_id=task["task_id"],
        status=task["status"],
        stage=task["stage"],
        progress=task["progress"],
    )


def _task_response(
    task: dict,
    request: Request,
) -> AnalysisTaskResponse:
    payload = dict(task)
    payload["result_url"] = (
        str(request.url_for("restore_case", case_id=task["case_id"]))
        if task["status"] == "succeeded"
        else None
    )
    return AnalysisTaskResponse.model_validate(payload)


@router.get(
    "/analysis-tasks/{task_id}",
    response_model=AnalysisTaskResponse,
    summary="查询MRI分析任务状态和阶段进度",
)
def get_analysis_task(
    task_id: str,
    request: Request,
    tasks: Annotated[
        AnalysisTaskRepository,
        Depends(get_analysis_task_repository),
    ],
) -> AnalysisTaskResponse:
    return _task_response(tasks.get(task_id), request)


@router.post(
    "/analysis-tasks/{task_id}/cancel",
    response_model=AnalysisTaskResponse,
    summary="请求取消MRI分析任务",
)
def cancel_analysis_task(
    task_id: str,
    request: Request,
    tasks: Annotated[
        AnalysisTaskRepository,
        Depends(get_analysis_task_repository),
    ],
) -> AnalysisTaskResponse:
    task = tasks.get(task_id)
    if task["status"] in {"queued", "running"}:
        task = tasks.update(
            task_id,
            status="cancel_requested",
            message="已请求取消；正在运行的推理将在当前阶段结束后停止",
        )
    return _task_response(task, request)


@router.post(
    "/report",
    response_model=ReportResponse,
    summary="生成MRI影像辅助报告",
)
def generate_report(
    payload: ReportRequest,
    service: Annotated[MedicalReportService, Depends(get_report_service)],
) -> ReportResponse:
    generated = service.generate(payload.case_id)
    return ReportResponse(
        case_id=payload.case_id,
        status="report_ready",
        report=generated.content,
        requires_human_review=True,
    )


@router.get(
    "/cases/{case_id}",
    response_model=CaseRestoreResponse,
    summary="读取病例状态和分析产物",
)
def restore_case(
    case_id: str,
    request: Request,
    repository: Annotated[CaseRepository, Depends(get_case_repository)],
    tasks: Annotated[
        AnalysisTaskRepository,
        Depends(get_analysis_task_repository),
    ],
) -> CaseRestoreResponse:
    paths = repository.require_case(case_id)
    status = repository.read_status(case_id)
    raw_modalities = status.get("modalities", {})
    modalities: dict[str, str] = {}
    for modality in MRIModality:
        filename = raw_modalities.get(modality.value)
        if not isinstance(filename, str):
            continue
        target = (paths.raw / filename).resolve()
        if target.is_file() and target.parent == paths.raw.resolve():
            modalities[modality.value] = str(
                request.url_for(
                    "download_modality",
                    case_id=paths.case_id,
                    modality=modality.value,
                )
            )

    mask = None
    if paths.mask.is_file():
        mask = MaskArtifact(
            filename=paths.mask.name,
            download_url=str(request.url_for("download_mask", case_id=paths.case_id)),
        )
    features = repository.load_features(case_id)
    tumor_metrics = TumorMetrics.model_validate(features) if features else None
    report = (
        paths.report.read_text(encoding="utf-8")
        if paths.report.is_file()
        else None
    )
    task = tasks.latest_for_case(paths.case_id)
    return CaseRestoreResponse(
        case_id=paths.case_id,
        status=str(status.get("status", "uploaded")),
        modalities=modalities,
        mask=mask,
        tumor_metrics=tumor_metrics,
        report=report,
        analysis_task=_task_response(task, request) if task else None,
    )


@router.get(
    "/cases/{case_id}/modalities/{modality}",
    response_class=FileResponse,
    name="download_modality",
    summary="下载病例原始MRI模态",
)
def download_modality(
    case_id: str,
    modality: str,
    repository: Annotated[CaseRepository, Depends(get_case_repository)],
) -> FileResponse:
    try:
        normalized_modality = MRIModality(modality)
    except ValueError as exc:
        from backend.app.services.errors import InvalidUploadError

        raise InvalidUploadError("不支持的MRI模态") from exc
    paths = repository.require_case(case_id)
    status = repository.read_status(case_id)
    filename = status.get("modalities", {}).get(normalized_modality.value)
    if not isinstance(filename, str):
        from backend.app.services.errors import CaseStateError

        raise CaseStateError("病例中不存在该MRI模态")
    target = (paths.raw / filename).resolve()
    if target.parent != paths.raw.resolve() or not target.is_file():
        from backend.app.services.errors import CaseStateError

        raise CaseStateError("MRI模态文件不存在")
    return FileResponse(
        path=target,
        media_type="application/gzip",
        filename=target.name,
    )


@router.post(
    "/report/edit",
    response_model=ReportEditResponse,
    summary="生成待医生确认的报告修改建议",
)
def propose_report_edit(
    payload: ReportEditRequest,
    service: Annotated[
        MedicalReportEditingService, Depends(get_report_editing_service)
    ],
) -> ReportEditResponse:
    proposal = service.propose(payload.case_id, payload.instruction)
    return ReportEditResponse(
        case_id=payload.case_id,
        status="edit_proposed",
        suggestion_id=proposal["suggestion_id"],
        current_report=proposal["current_report"],
        proposed_report=proposal["proposed_report"],
        change_summary=proposal["change_summary"],
        protected_metrics=proposal["protected_metrics"],
    )


@router.post(
    "/report/apply",
    response_model=ReportApplyResponse,
    summary="应用医生确认的报告修改建议",
)
def apply_report_edit(
    payload: ReportApplyRequest,
    service: Annotated[
        MedicalReportEditingService, Depends(get_report_editing_service)
    ],
) -> ReportApplyResponse:
    result = service.apply(payload.case_id, payload.suggestion_id)
    return ReportApplyResponse(
        case_id=payload.case_id,
        status="report_updated",
        revision_id=result["revision_id"],
        report=result["report"],
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="调用Qwen MRI Assistant Agent",
)
def chat_with_agent(
    payload: ChatRequest,
    service: Annotated[MedicalAgentChatService, Depends(get_chat_service)],
) -> ChatResponse:
    response = service.chat(
        question=payload.question,
        case_id=payload.case_id,
    )
    return ChatResponse(
        run_id=response.run_id,
        case_id=payload.case_id,
        intent=response.intent,
        tool_name=response.tool_name,
        answer=response.answer,
        citations=list(response.citations),
        safety_warnings=list(response.safety_warnings),
        requires_human_review=response.requires_human_review,
    )


@router.get(
    "/cases/{case_id}/mask",
    response_class=FileResponse,
    name="download_mask",
    summary="下载BraTS标签空间分割mask",
)
def download_mask(
    case_id: str,
    repository: Annotated[CaseRepository, Depends(get_case_repository)],
) -> FileResponse:
    paths = repository.require_case(case_id)
    if not paths.mask.is_file():
        from backend.app.services.errors import CaseStateError

        raise CaseStateError("该病例尚未生成分割mask，请先调用/analyze")
    return FileResponse(
        path=paths.mask,
        media_type="application/gzip",
        filename=paths.mask.name,
    )
