"""MRI读取、检查和处理过程中使用的数据契约。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from data_process.constants import MRIModality


@dataclass(frozen=True, slots=True)
class MRISeries:
    """一个病例中的单个MRI模态。"""

    modality: MRIModality
    path: Path


@dataclass(frozen=True, slots=True)
class StudyManifest:
    """去标识化的多模态MRI病例清单。"""

    case_id: str
    series: tuple[MRISeries, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    def by_modality(self) -> dict[MRIModality, MRISeries]:
        """以模态为键返回影像序列。"""

        return {item.modality: item for item in self.series}


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    """三维影像的尺寸和物理空间信息。"""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    affine: NDArray[np.float64] = field(repr=False, compare=False)


@dataclass(slots=True)
class NiftiVolume:
    """已读取的三维NIfTI体数据及其空间信息。"""

    path: Path
    data: NDArray[np.float32]
    geometry: ImageGeometry
    header: Any = field(repr=False)


@dataclass(slots=True)
class LoadedStudy:
    """完成四模态读取和几何检查的病例。"""

    case_id: str
    volumes: dict[MRIModality, NiftiVolume]

    def stacked_data(self) -> NDArray[np.float32]:
        """按T1、T1ce、T2、FLAIR顺序堆叠为(C, X, Y, Z)。"""

        from data_process.constants import REQUIRED_MODALITIES

        return np.stack(
            [self.volumes[modality].data for modality in REQUIRED_MODALITIES],
            axis=0,
        ).astype(np.float32, copy=False)

    @property
    def reference_geometry(self) -> ImageGeometry:
        """返回T1模态的参考几何信息。"""

        return self.volumes[MRIModality.T1].geometry


@dataclass(frozen=True, slots=True)
class ProcessedStudy:
    """一个病例完成处理后生成的文件清单。"""

    case_id: str
    output_dir: Path
    modality_files: dict[MRIModality, Path]
    metadata_path: Path
