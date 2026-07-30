"""FastAPI请求与响应数据契约。"""

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

__all__ = [
    "AnalyzeRequest",
    "AnalyzeResponse",
    "ChatRequest",
    "ChatResponse",
    "MaskArtifact",
    "ReportRequest",
    "ReportResponse",
    "TumorMetrics",
    "UploadResponse",
]
