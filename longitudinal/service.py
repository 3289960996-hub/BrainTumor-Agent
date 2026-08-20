"""Deterministic multi-timepoint MRI comparison service."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from backend.app.schemas.imaging import TumorMetrics
from backend.app.services.errors import CaseStateError
from backend.app.services.storage import CaseRepository
from longitudinal.registration import SpatialInputs, build_spatial_comparison
from longitudinal.schemas import LongitudinalComparison, MetricChange
from longitudinal.storage import ComparisonRepository

PHYSICAL_METRICS = (
    ("tumor_volume", "Whole Tumor体积", "cm3"),
    ("tumor_core_volume", "Tumor Core体积", "cm3"),
    ("enhancing_volume", "Enhancing Tumor体积", "cm3"),
    ("edema_volume", "水肿区域体积", "cm3"),
    ("max_diameter", "三维最大径", "mm"),
)
RATIO_METRICS = (
    ("tumor_core_ratio", "Tumor Core/Whole Tumor", "%"),
    ("enhancing_ratio", "Enhancing Tumor/Whole Tumor", "%"),
    ("edema_ratio", "水肿区域/Whole Tumor", "%"),
)
ComparisonProgress = Callable[[str, int, str], None]


class ComparisonCancellationRequested(Exception):
    """Raised when an asynchronous comparison is cancelled at a stage boundary."""


class LongitudinalComparisonService:
    """Compare existing quantitative results without diagnostic inference."""

    def __init__(
        self,
        cases: CaseRepository,
        comparisons: ComparisonRepository,
    ) -> None:
        self.cases = cases
        self.comparisons = comparisons

    def create(
        self,
        *,
        patient_group_id: str,
        baseline_case_id: str,
        followup_case_id: str,
        baseline_study_date: date,
        followup_study_date: date,
        progress: ComparisonProgress | None = None,
    ) -> dict[str, Any]:
        _report(progress, "validating", 5, "正在校验病例配对和检查日期")
        baseline, followup = self.validate_request(
            baseline_case_id=baseline_case_id,
            followup_case_id=followup_case_id,
            baseline_study_date=baseline_study_date,
            followup_study_date=followup_study_date,
        )

        _report(progress, "quantitative_comparison", 12, "正在计算确定性定量变化")
        comparison_id = _comparison_id(
            patient_group_id,
            baseline_case_id,
            followup_case_id,
            baseline_study_date,
            followup_study_date,
        )
        spatial = self._build_spatial(
            comparison_id,
            baseline_case_id,
            followup_case_id,
            progress,
        )
        result = LongitudinalComparison(
            comparison_id=comparison_id,
            patient_group_id=patient_group_id,
            baseline_case_id=baseline_case_id,
            followup_case_id=followup_case_id,
            baseline_study_date=baseline_study_date.isoformat(),
            followup_study_date=followup_study_date.isoformat(),
            interval_days=(followup_study_date - baseline_study_date).days,
            metrics=tuple(_metric_changes(baseline, followup)),
            baseline_location=baseline.location,
            followup_location=followup.location,
            location_consistent=_normalize_location(baseline.location)
            == _normalize_location(followup.location),
            created_at=datetime.now(UTC).isoformat(),
            spatial_comparison_available=spatial.get("status") == "quality_passed",
            spatial_comparison=spatial,
        )
        payload = result.to_dict()
        _report(progress, "saving", 96, "正在保存空间对比结果")
        self.comparisons.save(comparison_id, payload)
        return payload

    def validate_request(
        self,
        *,
        baseline_case_id: str,
        followup_case_id: str,
        baseline_study_date: date,
        followup_study_date: date,
    ) -> tuple[TumorMetrics, TumorMetrics]:
        if baseline_case_id == followup_case_id:
            raise CaseStateError("基线病例和随访病例不能是同一个病例")
        if followup_study_date <= baseline_study_date:
            raise CaseStateError("随访检查日期必须晚于基线检查日期")
        baseline = self._load_metrics(baseline_case_id, "基线")
        followup = self._load_metrics(followup_case_id, "随访")
        return baseline, followup

    def get(self, comparison_id: str) -> dict[str, Any]:
        return self.comparisons.get(comparison_id)

    def _load_metrics(self, case_id: str, role: str) -> TumorMetrics:
        self.cases.require_case(case_id)
        payload = self.cases.load_features(case_id)
        if payload is None:
            raise CaseStateError(f"{role}病例尚未完成定量分析：{case_id}")
        try:
            return TumorMetrics.model_validate(payload)
        except ValueError as exc:
            raise CaseStateError(f"{role}病例定量指标不完整：{case_id}") from exc

    def _build_spatial(
        self,
        comparison_id: str,
        baseline_case_id: str,
        followup_case_id: str,
        progress: ComparisonProgress | None,
    ) -> dict[str, Any]:
        try:
            baseline_paths = self.cases.require_case(baseline_case_id)
            followup_paths = self.cases.require_case(followup_case_id)
            inputs = SpatialInputs(
                fixed_t1ce=_modality_path(self.cases, baseline_case_id, "t1ce"),
                moving_t1ce=_modality_path(self.cases, followup_case_id, "t1ce"),
                fixed_mask=baseline_paths.mask,
                moving_mask=followup_paths.mask,
            )
            return build_spatial_comparison(
                inputs,
                self.comparisons.artifact_dir(comparison_id),
                progress=progress,
            )
        except FileNotFoundError as exc:
            return {
                "status": "unavailable",
                "method": "SimpleITK Euler3D rigid registration on T1ce",
                "unavailable_reason": str(exc),
                "quality": None,
                "changes": {},
                "artifacts": {},
            }
        except (OSError, RuntimeError, ValueError):
            return {
                "status": "unavailable",
                "method": "SimpleITK Euler3D rigid registration on T1ce",
                "unavailable_reason": "空间配准执行失败，请检查影像几何和内容",
                "quality": None,
                "changes": {},
                "artifacts": {},
            }


def _comparison_id(
    patient_group_id: str,
    baseline_case_id: str,
    followup_case_id: str,
    baseline_study_date: date,
    followup_study_date: date,
) -> str:
    source = "|".join(
        (
            patient_group_id,
            baseline_case_id,
            followup_case_id,
            baseline_study_date.isoformat(),
            followup_study_date.isoformat(),
        )
    )
    return "comparison-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]


def comparison_id_for(
    patient_group_id: str,
    baseline_case_id: str,
    followup_case_id: str,
    baseline_study_date: date,
    followup_study_date: date,
) -> str:
    return _comparison_id(
        patient_group_id,
        baseline_case_id,
        followup_case_id,
        baseline_study_date,
        followup_study_date,
    )


def _metric_changes(
    baseline: TumorMetrics,
    followup: TumorMetrics,
) -> list[MetricChange]:
    changes: list[MetricChange] = []
    for key, label, unit in PHYSICAL_METRICS:
        baseline_value = float(getattr(baseline, key))
        followup_value = float(getattr(followup, key))
        absolute = _round(followup_value - baseline_value)
        percent = (
            None
            if baseline_value == 0
            else _round((followup_value - baseline_value) / baseline_value * 100)
        )
        changes.append(
            MetricChange(
                key=key,
                label=label,
                unit=unit,
                baseline=_round(baseline_value),
                followup=_round(followup_value),
                absolute_change=absolute,
                percent_change=percent,
                percentage_point_change=None,
                direction=_direction(absolute),
            )
        )
    for key, label, unit in RATIO_METRICS:
        baseline_value = float(getattr(baseline, key)) * 100
        followup_value = float(getattr(followup, key)) * 100
        points = _round(followup_value - baseline_value)
        changes.append(
            MetricChange(
                key=key,
                label=label,
                unit=unit,
                baseline=_round(baseline_value),
                followup=_round(followup_value),
                absolute_change=points,
                percent_change=None,
                percentage_point_change=points,
                direction=_direction(points),
            )
        )
    return changes


def _direction(value: float) -> str:
    if value > 0:
        return "increased"
    if value < 0:
        return "decreased"
    return "unchanged"


def _normalize_location(value: str) -> str:
    return " ".join(value.lower().split())


def _round(value: float) -> float:
    return round(value, 4)


def _report(
    progress: ComparisonProgress | None,
    stage: str,
    percent: int,
    message: str,
) -> None:
    if progress is not None:
        progress(stage, percent, message)


def _modality_path(
    repository: CaseRepository,
    case_id: str,
    modality: str,
) -> Path:
    paths = repository.require_case(case_id)
    filename = repository.read_status(case_id).get("modalities", {}).get(modality)
    if not isinstance(filename, str):
        raise FileNotFoundError(f"病例缺少{modality}模态")
    target = (paths.raw / filename).resolve()
    if target.parent != paths.raw.resolve() or not target.is_file():
        raise FileNotFoundError(f"病例{modality}模态文件不存在")
    return target
