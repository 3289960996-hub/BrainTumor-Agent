"""MRI上传、分析、报告和Agent问答接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from backend.app.schemas.imaging import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    MaskArtifact,
    ReportRequest,
    ReportResponse,
    TumorMetrics,
    UploadResponse,
)
from backend.app.services.analysis import MRIAnalysisPipeline
from backend.app.services.chat import MedicalAgentChatService
from backend.app.services.dependencies import (
    get_analysis_pipeline,
    get_case_repository,
    get_chat_service,
    get_report_service,
    get_upload_service,
)
from backend.app.services.reporting import MedicalReportService
from backend.app.services.storage import CaseRepository
from backend.app.services.upload import MRIUploadService
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
    summary="执行MRI预处理、nnU-Net推理和指标计算",
)
def analyze_mri(
    payload: AnalyzeRequest,
    request: Request,
    pipeline: Annotated[MRIAnalysisPipeline, Depends(get_analysis_pipeline)],
) -> AnalyzeResponse:
    result = pipeline.analyze(payload.case_id)
    download_url = str(
        request.url_for("download_mask", case_id=result.case_id)
    )
    return AnalyzeResponse(
        case_id=result.case_id,
        status="analyzed",
        mask=MaskArtifact(
            filename=result.mask_path.name,
            download_url=download_url,
        ),
        tumor_metrics=TumorMetrics.model_validate(result.metrics),
    )


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
