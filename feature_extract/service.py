"""Feature extraction service boundary."""

from pathlib import Path

from feature_extract.schemas import ImagingFeatureSet


class FeatureExtractionService:
    """Calculate numeric features from a validated segmentation mask."""

    def __init__(self, extractor_version: str = "0.1.0") -> None:
        self.extractor_version = extractor_version

    def extract(self, case_id: str, segmentation_path: Path) -> ImagingFeatureSet:
        """Extract volume, location, and morphology features."""

        raise NotImplementedError(
            "Imaging feature extraction is not implemented in the project skeleton."
        )
