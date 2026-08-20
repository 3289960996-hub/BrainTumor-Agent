"""Rigid longitudinal registration and deterministic spatial change masks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

REGION_LABELS = {
    "wt": (1, 2, 4),
    "tc": (1, 4),
    "et": (4,),
}
SpatialProgress = Callable[[str, int, str], None]


@dataclass(frozen=True, slots=True)
class SpatialInputs:
    fixed_t1ce: Path
    moving_t1ce: Path
    fixed_mask: Path
    moving_mask: Path


def build_spatial_comparison(
    inputs: SpatialInputs,
    output_dir: Path,
    progress: SpatialProgress | None = None,
) -> dict[str, Any]:
    """Register follow-up T1ce to baseline space and create regional change masks."""

    _require_inputs(inputs)
    _report(progress, "loading_images", 25, "正在读取基线与随访影像")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed = sitk.ReadImage(str(inputs.fixed_t1ce), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(inputs.moving_t1ce), sitk.sitkFloat32)
    fixed_mask = sitk.ReadImage(str(inputs.fixed_mask), sitk.sitkUInt8)
    moving_mask = sitk.ReadImage(str(inputs.moving_mask), sitk.sitkUInt8)
    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        raise ValueError("空间对比仅支持三维MRI")

    _report(progress, "rigid_registration", 35, "正在执行T1ce刚性配准")
    initial_transform = _initial_rigid(fixed, moving)
    transform = _register_rigid(fixed, moving, initial_transform)
    registered_image = _resample_image(moving, fixed, transform)
    initial_image = _resample_image(moving, fixed, initial_transform)
    if _foreground_correlation(fixed, registered_image) < _foreground_correlation(
        fixed,
        initial_image,
    ):
        transform = initial_transform
        registered_image = initial_image
    _report(progress, "resampling_mask", 60, "正在重采样随访分割Mask")
    registered_mask = sitk.Resample(
        moving_mask,
        fixed_mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    registered_image_path = output_dir / "followup_t1ce_registered.nii.gz"
    registered_mask_path = output_dir / "followup_mask_registered.nii.gz"
    transform_path = output_dir / "rigid_transform.tfm"
    sitk.WriteImage(registered_image, str(registered_image_path), True)
    sitk.WriteImage(registered_mask, str(registered_mask_path), True)
    sitk.WriteTransform(transform, str(transform_path))

    _report(progress, "registration_quality", 70, "正在检查配准质量")
    quality = _quality_metrics(fixed, moving, registered_image, transform)
    baseline_labels = sitk.GetArrayFromImage(fixed_mask)
    followup_labels = sitk.GetArrayFromImage(registered_mask)
    voxel_volume_cm3 = float(np.prod(fixed_mask.GetSpacing()) / 1000.0)
    changes: dict[str, dict[str, float | int]] = {}
    artifacts: dict[str, str] = {
        "registered_followup_t1ce": registered_image_path.name,
        "registered_followup_mask": registered_mask_path.name,
        "rigid_transform": transform_path.name,
    }

    for index, (region, labels) in enumerate(REGION_LABELS.items()):
        _report(
            progress,
            "change_masks",
            78 + index * 5,
            f"正在计算{region.upper()}新增、持续和消退区域",
        )
        baseline_region = np.isin(baseline_labels, labels)
        followup_region = np.isin(followup_labels, labels)
        persistent = baseline_region & followup_region
        new = ~baseline_region & followup_region
        resolved = baseline_region & ~followup_region
        # 1=消退，2=持续，3=新增；三类互斥，便于前端直接着色。
        change = np.zeros(baseline_region.shape, dtype=np.uint8)
        change[resolved] = 1
        change[persistent] = 2
        change[new] = 3
        change_image = sitk.GetImageFromArray(change)
        change_image.CopyInformation(fixed_mask)
        change_path = output_dir / f"{region}_change.nii.gz"
        sitk.WriteImage(change_image, str(change_path), True)
        artifacts[f"{region}_change"] = change_path.name
        changes[region] = {
            "resolved_voxels": int(np.count_nonzero(resolved)),
            "persistent_voxels": int(np.count_nonzero(persistent)),
            "new_voxels": int(np.count_nonzero(new)),
            "resolved_volume_cm3": _round(np.count_nonzero(resolved) * voxel_volume_cm3),
            "persistent_volume_cm3": _round(
                np.count_nonzero(persistent) * voxel_volume_cm3
            ),
            "new_volume_cm3": _round(np.count_nonzero(new) * voxel_volume_cm3),
        }

    return {
        "status": "quality_passed" if quality["passed"] else "quality_failed",
        "method": "SimpleITK Euler3D rigid registration on T1ce",
        "quality": quality,
        "changes": changes,
        "artifacts": artifacts,
    }


def _report(
    progress: SpatialProgress | None,
    stage: str,
    percent: int,
    message: str,
) -> None:
    if progress is not None:
        progress(stage, percent, message)


def _initial_rigid(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )


def _register_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    initial: sitk.Transform,
) -> sitk.Transform:
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.15, seed=20260820)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=120,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    return registration.Execute(fixed, moving)


def _resample_image(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    return sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def _quality_metrics(
    fixed: sitk.Image,
    moving: sitk.Image,
    registered: sitk.Image,
    transform: sitk.Transform,
) -> dict[str, Any]:
    identity_moving = sitk.Resample(moving, fixed, sitk.Transform(3, sitk.sitkIdentity))
    before = _foreground_correlation(fixed, identity_moving)
    after = _foreground_correlation(fixed, registered)
    overlap = _foreground_dice(fixed, registered)
    rigid = _as_euler_transform(transform)
    parameters = rigid.GetParameters()
    rotation_degrees = [math.degrees(float(value)) for value in parameters[:3]]
    translation_mm = [float(value) for value in parameters[3:6]]
    max_rotation = max(abs(value) for value in rotation_degrees)
    translation_norm = math.sqrt(sum(value * value for value in translation_mm))
    passed = (
        after >= 0.35
        and after >= before - 0.02
        and overlap >= 0.70
        and max_rotation <= 30.0
        and translation_norm <= 80.0
    )
    warnings: list[str] = []
    if after < 0.35:
        warnings.append("配准后T1ce强度相关性偏低")
    if after < before - 0.02:
        warnings.append("配准后相似度未改善")
    if overlap < 0.70:
        warnings.append("配准后脑区重叠不足")
    if max_rotation > 30.0 or translation_norm > 80.0:
        warnings.append("刚性变换幅度异常")
    return {
        "passed": passed,
        "correlation_before": _round(before),
        "correlation_after": _round(after),
        "foreground_dice": _round(overlap),
        "rotation_degrees": [_round(value) for value in rotation_degrees],
        "translation_mm": [_round(value) for value in translation_mm],
        "warnings": warnings,
    }


def _foreground_correlation(first: sitk.Image, second: sitk.Image) -> float:
    left = sitk.GetArrayViewFromImage(first).ravel()
    right = sitk.GetArrayViewFromImage(second).ravel()
    foreground = (left != 0) & (right != 0) & np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(foreground) < 32:
        return 0.0
    left_values = left[foreground].astype(np.float64, copy=False)
    right_values = right[foreground].astype(np.float64, copy=False)
    if np.std(left_values) == 0 or np.std(right_values) == 0:
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _foreground_dice(first: sitk.Image, second: sitk.Image) -> float:
    left = sitk.GetArrayViewFromImage(first) != 0
    right = sitk.GetArrayViewFromImage(second) != 0
    denominator = int(np.count_nonzero(left) + np.count_nonzero(right))
    if denominator == 0:
        return 0.0
    return float(2 * np.count_nonzero(left & right) / denominator)


def _require_inputs(inputs: SpatialInputs) -> None:
    paths = (
        inputs.fixed_t1ce,
        inputs.moving_t1ce,
        inputs.fixed_mask,
        inputs.moving_mask,
    )
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("空间对比缺少文件：" + "、".join(missing))


def _round(value: float) -> float:
    return round(float(value), 4)


def _as_euler_transform(transform: sitk.Transform) -> sitk.Euler3DTransform:
    current = transform
    while current.GetName() == "CompositeTransform":
        composite = sitk.CompositeTransform(current)
        if composite.GetNumberOfTransforms() == 0:
            raise ValueError("配准没有生成有效的刚性变换")
        current = composite.GetBackTransform()
    return sitk.Euler3DTransform(current)
