"""Segmentation request and result contracts."""

from dataclasses import dataclass
from pathlib import Path

from data_process.schemas import StudyManifest


@dataclass(frozen=True, slots=True)
class SegmentationRequest:
    """Input required to start one segmentation inference."""

    study: StudyManifest
    model_version: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Artifacts produced by a successful segmentation inference."""

    case_id: str
    model_version: str
    segmentation_path: Path
    qc_status: str
    warnings: tuple[str, ...] = ()
