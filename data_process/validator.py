"""MRI清单、尺寸、spacing和物理空间一致性检查。"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from data_process.constants import REQUIRED_MODALITIES, MRIModality
from data_process.exceptions import GeometryMismatchError, ManifestValidationError
from data_process.schemas import NiftiVolume, StudyManifest


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """数据检查结果。"""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def validate_manifest(manifest: StudyManifest) -> ValidationResult:
    """检查模态完整性、重复项和文件是否存在。"""

    errors: list[str] = []
    modality_map = manifest.by_modality()

    if len(modality_map) != len(manifest.series):
        errors.append("病例中存在重复MRI模态")

    missing = [modality.value for modality in REQUIRED_MODALITIES if modality not in modality_map]
    if missing:
        errors.append(f"缺少必需模态：{', '.join(missing)}")

    for item in manifest.series:
        if not item.path.is_file():
            errors.append(f"输入文件不存在：{item.path}")

    return ValidationResult(valid=not errors, errors=tuple(errors))


def ensure_valid_manifest(manifest: StudyManifest) -> None:
    """清单不合法时抛出包含全部错误信息的异常。"""

    result = validate_manifest(manifest)
    if not result.valid:
        raise ManifestValidationError("；".join(result.errors))


def check_image_sizes(
    volumes: Mapping[MRIModality, NiftiVolume],
) -> ValidationResult:
    """检查四模态三维尺寸是否完全一致。"""

    errors: list[str] = []
    reference = volumes[REQUIRED_MODALITIES[0]].geometry.shape
    for modality in REQUIRED_MODALITIES[1:]:
        current = volumes[modality].geometry.shape
        if current != reference:
            errors.append(
                f"{modality.value}尺寸{current}与T1参考尺寸{reference}不一致"
            )
    return ValidationResult(valid=not errors, errors=tuple(errors))


def check_spacings(
    volumes: Mapping[MRIModality, NiftiVolume],
    tolerance: float = 1e-5,
) -> ValidationResult:
    """检查四模态spacing是否在指定绝对误差内一致。"""

    errors: list[str] = []
    reference = volumes[REQUIRED_MODALITIES[0]].geometry.spacing
    for modality in REQUIRED_MODALITIES[1:]:
        current = volumes[modality].geometry.spacing
        if not np.allclose(current, reference, rtol=0.0, atol=tolerance):
            errors.append(
                f"{modality.value} spacing {current}与T1参考spacing {reference}不一致"
            )
    return ValidationResult(valid=not errors, errors=tuple(errors))


def validate_multimodal_geometry(
    volumes: Mapping[MRIModality, NiftiVolume],
    spacing_tolerance: float = 1e-5,
    affine_tolerance: float = 1e-4,
) -> ValidationResult:
    """综合检查尺寸、spacing、affine、origin和direction。

    BraTS四模态理论上已经完成配准。仅尺寸和spacing一致仍不足以证明物理
    空间一致，因此同时检查affine以及SimpleITK读取到的origin/direction。
    """

    missing = [modality.value for modality in REQUIRED_MODALITIES if modality not in volumes]
    if missing:
        return ValidationResult(
            valid=False,
            errors=(f"缺少已加载模态：{', '.join(missing)}",),
        )

    errors = list(check_image_sizes(volumes).errors)
    errors.extend(check_spacings(volumes, tolerance=spacing_tolerance).errors)

    reference = volumes[REQUIRED_MODALITIES[0]].geometry
    for modality in REQUIRED_MODALITIES[1:]:
        current = volumes[modality].geometry
        if not np.allclose(
            current.affine,
            reference.affine,
            rtol=0.0,
            atol=affine_tolerance,
        ):
            errors.append(f"{modality.value} affine与T1不一致")
        if not np.allclose(
            current.origin,
            reference.origin,
            rtol=0.0,
            atol=affine_tolerance,
        ):
            errors.append(f"{modality.value} origin与T1不一致")
        if not np.allclose(
            current.direction,
            reference.direction,
            rtol=0.0,
            atol=affine_tolerance,
        ):
            errors.append(f"{modality.value} direction与T1不一致")

    return ValidationResult(valid=not errors, errors=tuple(errors))


def ensure_compatible_geometry(
    volumes: Mapping[MRIModality, NiftiVolume],
    spacing_tolerance: float = 1e-5,
    affine_tolerance: float = 1e-4,
) -> None:
    """几何不一致时拒绝继续归一化和保存。"""

    result = validate_multimodal_geometry(
        volumes,
        spacing_tolerance=spacing_tolerance,
        affine_tolerance=affine_tolerance,
    )
    if not result.valid:
        raise GeometryMismatchError("；".join(result.errors))
