"""MRI上传、分析、报告和Agent问答的API模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


class StrictModel(BaseModel):
    """拒绝未声明字段，避免调用方误以为参数已经生效。"""

    model_config = ConfigDict(extra="forbid")


class UploadResponse(StrictModel):
    case_id: str
    status: Literal["uploaded"]
    modalities: dict[str, str]


class AnalyzeRequest(StrictModel):
    case_id: str = Field(pattern=CASE_ID_PATTERN)


class TumorMetrics(StrictModel):
    tumor_volume: float
    tumor_core_volume: float
    enhancing_volume: float
    max_diameter: float
    edema: bool
    location: str
    edema_volume: float
    tumor_core_ratio: float
    enhancing_ratio: float
    edema_ratio: float


class MaskArtifact(StrictModel):
    filename: str
    download_url: str
    label_space: Literal["brats"] = "brats"


class AnalyzeResponse(StrictModel):
    case_id: str
    status: Literal["analyzed"]
    mask: MaskArtifact
    tumor_metrics: TumorMetrics


class ReportRequest(StrictModel):
    case_id: str = Field(pattern=CASE_ID_PATTERN)


class ReportResponse(StrictModel):
    case_id: str
    status: Literal["report_ready"]
    report: str
    requires_human_review: bool = True


class ChatRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    case_id: str | None = Field(default=None, pattern=CASE_ID_PATTERN)


class ChatResponse(StrictModel):
    run_id: str
    case_id: str | None
    intent: str
    tool_name: str | None
    answer: str
    citations: list[str]
    safety_warnings: list[str]
    requires_human_review: bool = True
