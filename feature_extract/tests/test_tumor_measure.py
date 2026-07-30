"""肿瘤定量测量模块测试。"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from feature_extract.tumor_measure import (
    calculate_max_diameter_mm,
    labels_to_tumor_regions,
    measure_tumor,
    save_measurement,
)


def _save_mask(
    path: Path,
    data: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    image = nib.Nifti1Image(data.astype(np.int16), np.diag([*spacing, 1.0]))
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


def test_brats_region_definitions() -> None:
    labels = np.asarray([0, 1, 2, 4], dtype=np.int16).reshape(2, 2, 1)

    regions = labels_to_tumor_regions(labels, "brats")

    assert int(regions.whole_tumor.sum()) == 3
    assert int(regions.tumor_core.sum()) == 2
    assert int(regions.enhancing_tumor.sum()) == 1
    assert int(regions.edema.sum()) == 1


def test_nnunet_region_definitions() -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int16).reshape(2, 2, 1)

    regions = labels_to_tumor_regions(labels, "nnunet")

    assert int(regions.whole_tumor.sum()) == 3
    assert int(regions.tumor_core.sum()) == 2
    assert int(regions.enhancing_tumor.sum()) == 1
    assert int(regions.edema.sum()) == 1


def test_max_diameter_uses_physical_spacing() -> None:
    mask = np.zeros((6, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    mask[4, 1, 1] = True
    affine = np.diag([2.0, 1.0, 1.0, 1.0])

    diameter = calculate_max_diameter_mm(mask, affine)

    assert diameter == pytest.approx(6.0)


def test_measure_tumor_returns_volumes_ratios_and_agent_payload(tmp_path: Path) -> None:
    labels = np.zeros((12, 12, 12), dtype=np.int16)
    labels[1:3, 8:10, 6:8] = 2  # 水肿：8 voxels
    labels[3:4, 8:10, 6:8] = 1  # 非增强核心：4 voxels
    labels[4:5, 8:10, 6:8] = 4  # 增强肿瘤：4 voxels
    mask_path = _save_mask(tmp_path / "case.nii.gz", labels, (1.0, 1.0, 2.0))

    result = measure_tumor(mask_path, label_space="brats")

    assert result.tumor_volume == pytest.approx(0.032)
    assert result.tumor_core_volume == pytest.approx(0.016)
    assert result.enhancing_volume == pytest.approx(0.008)
    assert result.edema_volume == pytest.approx(0.016)
    assert result.tumor_core_ratio == 0.5
    assert result.enhancing_ratio == 0.25
    assert result.edema_ratio == 0.5
    assert result.max_diameter > 0
    assert result.edema is True
    assert result.location == "left frontal"
    assert set(result.to_summary_dict()) == {
        "tumor_volume",
        "tumor_core_volume",
        "enhancing_volume",
        "max_diameter",
        "edema",
        "location",
    }


def test_empty_mask_returns_zero_measurements(tmp_path: Path) -> None:
    mask_path = _save_mask(
        tmp_path / "empty.nii.gz",
        np.zeros((6, 6, 6), dtype=np.int16),
    )

    result = measure_tumor(mask_path)

    assert result.tumor_volume == 0.0
    assert result.tumor_core_volume == 0.0
    assert result.enhancing_volume == 0.0
    assert result.edema_volume == 0.0
    assert result.max_diameter == 0.0
    assert result.edema is False
    assert result.location == "none"
    assert result.tumor_core_ratio == 0.0
    assert result.enhancing_ratio == 0.0
    assert result.edema_ratio == 0.0


def test_save_measurement_supports_summary_schema(tmp_path: Path) -> None:
    labels = np.zeros((6, 6, 6), dtype=np.int16)
    labels[1:3, 4:6, 3:5] = 4
    mask_path = _save_mask(tmp_path / "case.nii.gz", labels)
    result = measure_tumor(mask_path)

    output_path = save_measurement(
        result,
        tmp_path / "measurement.json",
        include_ratios=False,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload == result.to_summary_dict()
