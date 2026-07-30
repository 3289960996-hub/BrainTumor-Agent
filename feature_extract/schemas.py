"""Structured outputs from deterministic feature extraction."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RegionMeasurement:
    """Quantitative measurements for one derived tumor region."""

    region: str
    volume_ml: float
    voxel_count: int


@dataclass(frozen=True, slots=True)
class ImagingFeatureSet:
    """Versioned quantitative imaging features for one case."""

    case_id: str
    extractor_version: str
    regions: tuple[RegionMeasurement, ...]
    location: dict[str, Any] = field(default_factory=dict)
    morphology: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
