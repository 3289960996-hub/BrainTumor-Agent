"""MRI上传、分析、报告和Agent问答的API模型。"""

from __future__ import annotations

from datetime import date, datetime
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


AnalysisTaskStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancel_requested",
    "cancelled",
]


class AnalyzeResponse(StrictModel):
    case_id: str
    task_id: str
    status: AnalysisTaskStatus
    stage: str
    progress: int = Field(ge=0, le=100)


class AnalysisTaskResponse(StrictModel):
    task_id: str
    case_id: str
    status: AnalysisTaskStatus
    stage: str
    progress: int = Field(ge=0, le=100)
    message: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    mask_filename: str | None = None
    result_url: str | None = None


class CaseRestoreResponse(StrictModel):
    """病例恢复所需的持久化状态和产物地址。"""

    case_id: str
    status: str
    modalities: dict[str, str]
    mask: MaskArtifact | None = None
    tumor_metrics: TumorMetrics | None = None
    report: str | None = None
    analysis_task: AnalysisTaskResponse | None = None


class CaseSummary(StrictModel):
    case_id: str
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    has_metrics: bool
    has_report: bool


class CaseListResponse(StrictModel):
    cases: list[CaseSummary]


class ComparisonCreateRequest(StrictModel):
    patient_group_id: str = Field(pattern=CASE_ID_PATTERN)
    baseline_case_id: str = Field(pattern=CASE_ID_PATTERN)
    followup_case_id: str = Field(pattern=CASE_ID_PATTERN)
    baseline_study_date: date
    followup_study_date: date


ComparisonTaskStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancel_requested",
    "cancelled",
]


class ComparisonTaskResponse(StrictModel):
    task_id: str
    comparison_id: str
    status: ComparisonTaskStatus
    stage: str
    progress: int = Field(ge=0, le=100)
    message: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    result_url: str | None = None


class ComparisonMetricChange(StrictModel):
    key: str
    label: str
    unit: str
    baseline: float
    followup: float
    absolute_change: float
    percent_change: float | None
    percentage_point_change: float | None
    direction: Literal["increased", "decreased", "unchanged"]


class RegistrationQuality(StrictModel):
    passed: bool
    correlation_before: float
    correlation_after: float
    foreground_dice: float
    rotation_degrees: list[float]
    translation_mm: list[float]
    warnings: list[str]


class SpatialRegionChange(StrictModel):
    resolved_voxels: int = Field(ge=0)
    persistent_voxels: int = Field(ge=0)
    new_voxels: int = Field(ge=0)
    resolved_volume_cm3: float = Field(ge=0)
    persistent_volume_cm3: float = Field(ge=0)
    new_volume_cm3: float = Field(ge=0)


class SpatialComparison(StrictModel):
    status: Literal["quality_passed", "quality_failed", "unavailable"]
    method: str
    unavailable_reason: str | None = None
    quality: RegistrationQuality | None = None
    changes: dict[str, SpatialRegionChange]
    artifacts: dict[str, str]
    baseline_t1ce_url: str | None = None


class ComparisonResponse(StrictModel):
    comparison_id: str
    status: Literal["comparison_ready"] = "comparison_ready"
    comparison_version: int
    patient_group_id: str
    baseline_case_id: str
    followup_case_id: str
    baseline_study_date: date
    followup_study_date: date
    interval_days: int = Field(gt=0)
    metrics: list[ComparisonMetricChange]
    baseline_location: str
    followup_location: str
    location_consistent: bool
    spatial_comparison_available: bool
    spatial_comparison: SpatialComparison | None = None
    created_at: datetime
    requires_human_review: bool = True


class ReportRequest(StrictModel):
    case_id: str = Field(pattern=CASE_ID_PATTERN)


class ReportResponse(StrictModel):
    case_id: str
    status: Literal["report_ready"]
    report: str
    requires_human_review: bool = True


class ReportEditRequest(StrictModel):
    """Doctor instruction for a report edit proposal."""

    case_id: str = Field(pattern=CASE_ID_PATTERN)
    instruction: str = Field(min_length=1, max_length=2000)


class ReportEditResponse(StrictModel):
    """A non-destructive report edit proposal awaiting doctor confirmation."""

    case_id: str
    status: Literal["edit_proposed"]
    suggestion_id: str
    current_report: str
    proposed_report: str
    change_summary: list[str]
    protected_metrics: dict[str, float | bool | str]
    requires_confirmation: bool = True


class ReportApplyRequest(StrictModel):
    """Apply a previously reviewed report proposal."""

    case_id: str = Field(pattern=CASE_ID_PATTERN)
    suggestion_id: str = Field(min_length=8, max_length=64)


class ReportApplyResponse(StrictModel):
    """Result of saving a confirmed report revision."""

    case_id: str
    status: Literal["report_updated"]
    revision_id: str
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
