"""Segmentation service contracts for nnU-Net inference."""

from segmentation.schemas import SegmentationRequest, SegmentationResult
from segmentation.service import NnUNetSegmentationService

__all__ = [
    "NnUNetSegmentationService",
    "SegmentationRequest",
    "SegmentationResult",
]
