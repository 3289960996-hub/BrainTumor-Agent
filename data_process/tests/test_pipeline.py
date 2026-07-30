"""使用合成NIfTI验证数据处理模块，不需要真实BraTS数据或GPU。"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from data_process.constants import NNUNET_CHANNELS, REQUIRED_MODALITIES, MRIModality
from data_process.exceptions import GeometryMismatchError
from data_process.io import read_nifti
from data_process.loader import load_brats_case
from data_process.normalization import normalize_multimodal
from data_process.processor import BraTSDataProcessor


def _write_case(
    case_dir: Path,
    case_id: str = "BraTS2021_00000",
    shape_overrides: dict[MRIModality, tuple[int, int, int]] | None = None,
    spacing_overrides: dict[MRIModality, tuple[float, float, float]] | None = None,
) -> None:
    """在临时目录创建四个具有非零脑区的合成NIfTI。"""

    case_dir.mkdir(parents=True, exist_ok=True)
    shape_overrides = shape_overrides or {}
    spacing_overrides = spacing_overrides or {}

    for index, modality in enumerate(REQUIRED_MODALITIES):
        shape = shape_overrides.get(modality, (8, 9, 10))
        spacing = spacing_overrides.get(modality, (1.0, 1.0, 1.0))
        affine = np.diag([*spacing, 1.0]).astype(np.float64)

        data = np.zeros(shape, dtype=np.float32)
        inner_shape = tuple(max(dimension - 2, 1) for dimension in shape)
        values = np.arange(np.prod(inner_shape), dtype=np.float32).reshape(inner_shape)
        data[1:-1, 1:-1, 1:-1] = values + 1.0 + index * 10.0

        image = nib.Nifti1Image(data, affine)
        nib.save(image, case_dir / f"{case_id}_{modality.value}.nii.gz")


def test_read_nifti_preserves_shape_and_spacing(tmp_path: Path) -> None:
    """Nibabel体素和SimpleITK几何信息应保持一致。"""

    case_dir = tmp_path / "case"
    _write_case(case_dir)

    volume = read_nifti(case_dir / "BraTS2021_00000_t1.nii.gz")

    assert volume.data.shape == (8, 9, 10)
    assert volume.data.dtype == np.float32
    assert volume.geometry.spacing == pytest.approx((1.0, 1.0, 1.0))


def test_load_and_normalize_four_modalities(tmp_path: Path) -> None:
    """四模态应按固定顺序堆叠，并在非零区域完成逐通道Z-score。"""

    case_dir = tmp_path / "case"
    _write_case(case_dir)
    study = load_brats_case(case_dir)

    stacked = study.stacked_data()
    normalized = normalize_multimodal(stacked)

    assert stacked.shape == (4, 8, 9, 10)
    assert normalized.shape == stacked.shape
    for channel_index in range(4):
        mask = stacked[channel_index] != 0
        assert normalized[channel_index][~mask] == pytest.approx(0.0)
        assert float(normalized[channel_index][mask].mean()) == pytest.approx(
            0.0, abs=1e-5
        )
        assert float(normalized[channel_index][mask].std()) == pytest.approx(
            1.0, abs=1e-5
        )


def test_processor_saves_nnunet_compatible_files(tmp_path: Path) -> None:
    """处理器应保存四个通道文件和可追溯元数据。"""

    case_dir = tmp_path / "case"
    output_root = tmp_path / "processed"
    _write_case(case_dir)

    result = BraTSDataProcessor().process_case(case_dir, output_root)

    assert result.metadata_path.is_file()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["shape"] == [8, 9, 10]
    assert metadata["spacing"] == pytest.approx([1.0, 1.0, 1.0])
    assert metadata["normalization"]["method"] == "monai.NormalizeIntensity"

    for modality in REQUIRED_MODALITIES:
        output_path = result.modality_files[modality]
        assert output_path.name == (
            f"BraTS2021_00000_{NNUNET_CHANNELS[modality]}.nii.gz"
        )
        assert output_path.is_file()
        assert read_nifti(output_path).data.shape == (8, 9, 10)


def test_size_mismatch_is_rejected(tmp_path: Path) -> None:
    """任一模态尺寸不一致时必须终止处理。"""

    case_dir = tmp_path / "case"
    _write_case(
        case_dir,
        shape_overrides={MRIModality.T2: (7, 9, 10)},
    )

    with pytest.raises(GeometryMismatchError, match="尺寸"):
        load_brats_case(case_dir)


def test_spacing_mismatch_is_rejected(tmp_path: Path) -> None:
    """任一模态spacing不一致时必须终止处理。"""

    case_dir = tmp_path / "case"
    _write_case(
        case_dir,
        spacing_overrides={MRIModality.FLAIR: (1.2, 1.0, 1.0)},
    )

    with pytest.raises(GeometryMismatchError, match="spacing"):
        load_brats_case(case_dir)
