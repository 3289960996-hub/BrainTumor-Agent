"""Structured longitudinal comparison results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class MetricChange:
    """One arithmetic change between baseline and follow-up measurements."""

    key: str
    label: str
    unit: str
    baseline: float
    followup: float
    absolute_change: float
    percent_change: float | None
    percentage_point_change: float | None
    direction: Literal["increased", "decreased", "unchanged"]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongitudinalComparison:
    """Versioned deterministic comparison between two analyzed cases."""

    comparison_id: str
    patient_group_id: str
    baseline_case_id: str
    followup_case_id: str
    baseline_study_date: str
    followup_study_date: str
    interval_days: int
    metrics: tuple[MetricChange, ...]
    baseline_location: str
    followup_location: str
    location_consistent: bool
    created_at: str
    comparison_version: int = 2
    spatial_comparison_available: bool = False
    spatial_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["metrics"] = [metric.to_dict() for metric in self.metrics]
        return payload
