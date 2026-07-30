"""MRI 分割结果分析器测试。"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from feature_extract.analyzer import (
    AnalysisError,
    analyze_mask,
    infer_location_from_atlas,
    save_analysis,
)


def _save_nifti(path: Path, data: np.ndarray, spacing: tuple[float, ...]) -> Path:
    affine = np.diag([*spacing, 1.0])
    image = nib.Nifti1Image(data.astype(np.int16), affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


def test_analyze_brats_mask_volume_ratio_location_and_edema(tmp_path: Path) -> None:
    mask = np.zeros((20, 20, 20), dtype=np.int16)
    mask[2:4, 15:17, 10:12] = 2
    mask[4:5, 15:17, 10:12] = 4
    mask_path = _save_nifti(tmp_path / "case.nii.gz", mask, (1.0, 1.0, 2.0))

    result = analyze_mask(mask_path, label_space="brats")

    assert result.tumor_volume == pytest.approx(0.024)
    assert result.location == "left frontal"
    assert result.enhancing_ratio == pytest.approx(1 / 3, abs=1e-4)
    assert result.edema is True
    assert set(result.to_dict()) == {
        "tumor_volume",
        "location",
        "enhancing_ratio",
        "edema",
    }


def test_analyze_nnunet_internal_labels(tmp_path: Path) -> None:
    mask = np.zeros((10, 10, 10), dtype=np.int16)
    mask[7:9, 4:6, 1:3] = 1
    mask[7:9, 4:6, 3:4] = 3
    mask_path = _save_nifti(tmp_path / "case.nii.gz", mask, (1.0, 1.0, 1.0))

    result = analyze_mask(mask_path, label_space="nnunet")

    assert result.tumor_volume == pytest.approx(0.012)
    assert result.location == "right temporal"
    assert result.enhancing_ratio == pytest.approx(1 / 3, abs=1e-4)
    assert result.edema is True


def test_empty_mask_has_zero_measurements(tmp_path: Path) -> None:
    mask_path = _save_nifti(
        tmp_path / "empty.nii.gz",
        np.zeros((8, 8, 8), dtype=np.int16),
        (1.0, 1.0, 1.0),
    )

    result = analyze_mask(mask_path)

    assert result.tumor_volume == 0.0
    assert result.location == "none"
    assert result.enhancing_ratio == 0.0
    assert result.edema is False


def test_atlas_overlap_takes_priority(tmp_path: Path) -> None:
    mask = np.zeros((8, 8, 8), dtype=np.int16)
    mask[1:4, 5:7, 4:6] = 2
    atlas = np.zeros_like(mask)
    atlas[1:3, 5:7, 4:6] = 7
    atlas[3:4, 5:7, 4:6] = 8
    mask_path = _save_nifti(tmp_path / "mask.nii.gz", mask, (1.0, 1.0, 1.0))
    atlas_path = _save_nifti(tmp_path / "atlas.nii.gz", atlas, (1.0, 1.0, 1.0))

    result = analyze_mask(
        mask_path,
        atlas_path=atlas_path,
        atlas_label_map={7: "left frontal", 8: "left parietal"},
    )

    assert result.location == "left frontal"


def test_atlas_geometry_mismatch_is_rejected(tmp_path: Path) -> None:
    mask = np.zeros((8, 8, 8), dtype=np.int16)
    mask[1:2, 1:2, 1:2] = 2
    mask_path = _save_nifti(tmp_path / "mask.nii.gz", mask, (1.0, 1.0, 1.0))
    atlas_path = _save_nifti(tmp_path / "atlas.nii.gz", mask, (2.0, 1.0, 1.0))
    image = nib.load(mask_path)

    with pytest.raises(AnalysisError, match="affine"):
        infer_location_from_atlas(
            whole_tumor=mask > 0,
            mask_affine=np.asarray(image.affine),
            atlas_path=atlas_path,
            atlas_label_map={2: "left frontal"},
        )


def test_save_analysis_writes_agent_ready_json(tmp_path: Path) -> None:
    mask = np.zeros((8, 8, 8), dtype=np.int16)
    mask[1:3, 6:8, 4:6] = 4
    mask_path = _save_nifti(tmp_path / "mask.nii.gz", mask, (1.0, 1.0, 1.0))
    result = analyze_mask(mask_path)

    output = save_analysis(result, tmp_path / "analysis.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == result.to_dict()
