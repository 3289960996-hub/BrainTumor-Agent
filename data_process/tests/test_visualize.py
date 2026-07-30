"""四模态MRI和seg mask可视化测试。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nibabel as nib
import numpy as np
import pytest

from data_process.constants import REQUIRED_MODALITIES
from data_process.exceptions import VisualizationError
from data_process.visualize import (
    MaskLabelSpace,
    SliceAxis,
    create_matplotlib_figure,
    create_radiology_viewer_figure,
    generate_visualizations,
    load_visualization_case,
    select_slice_index,
)


def _write_visualization_case(
    case_dir: Path,
    case_id: str = "BraTS2021_00001",
    mask_shape: tuple[int, int, int] = (8, 9, 6),
) -> None:
    """创建带四模态和BraTS标签mask的合成病例。"""

    case_dir.mkdir(parents=True, exist_ok=True)
    shape = (8, 9, 6)
    affine = np.eye(4, dtype=np.float64)

    for index, modality in enumerate(REQUIRED_MODALITIES):
        grid = np.indices(shape, dtype=np.float32).sum(axis=0)
        data = grid + index * 20.0
        data[[0, -1], :, :] = 0
        data[:, [0, -1], :] = 0
        image = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(image, case_dir / f"{case_id}_{modality.value}.nii.gz")

    mask = np.zeros(mask_shape, dtype=np.uint8)
    if mask_shape == shape:
        mask[2:6, 2:7, 3] = 2
        mask[3:5, 3:6, 3] = 1
        mask[3:5, 4:6, 3] = 4
        mask[3:5, 3:5, 2] = 2
    nib.save(
        nib.Nifti1Image(mask, affine),
        case_dir / f"{case_id}_seg.nii.gz",
    )


def test_segmentation_is_discovered_and_best_slice_selected(tmp_path: Path) -> None:
    """自动发现seg后，应选择肿瘤面积最大的轴位层。"""

    case_dir = tmp_path / "case"
    _write_visualization_case(case_dir)

    case = load_visualization_case(case_dir)

    assert case.segmentation is not None
    assert select_slice_index(case, SliceAxis.AXIAL) == 3


def test_matplotlib_contains_four_modality_windows(tmp_path: Path) -> None:
    """Matplotlib静态图必须包含四个模态窗口。"""

    case_dir = tmp_path / "case"
    _write_visualization_case(case_dir)
    case = load_visualization_case(case_dir)

    figure, selected_index = create_matplotlib_figure(case)

    assert selected_index == 3
    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes] == ["T1", "T1ce", "T2", "FLAIR"]


def test_png_and_plotly_html_are_generated(tmp_path: Path) -> None:
    """完整流程应生成静态PNG和带Plotly内容的HTML。"""

    case_dir = tmp_path / "case"
    output_dir = tmp_path / "visualizations"
    _write_visualization_case(case_dir)

    result = generate_visualizations(
        case_dir=case_dir,
        output_dir=output_dir,
        axis=SliceAxis.AXIAL,
        slice_step=2,
    )

    assert result.static_path.is_file()
    assert result.static_path.stat().st_size > 0
    assert result.interactive_path is not None
    assert result.interactive_path.is_file()
    html = result.interactive_path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "BraTS2021_00001" in html
    assert "AI Segmentation Review" in html
    assert "Enhancing Tumor" in html


def test_radiology_viewer_has_modality_and_region_controls(tmp_path: Path) -> None:
    """阅片器应包含四个模态底图和三个可独立控制的分割图层。"""

    case_dir = tmp_path / "case"
    _write_visualization_case(case_dir)
    case = load_visualization_case(case_dir)

    figure = create_radiology_viewer_figure(
        case,
        axis=SliceAxis.AXIAL,
        slice_step=2,
    )

    assert len(figure.data) == 7
    assert [trace.name for trace in figure.data[:4]] == [
        "T1",
        "T1ce",
        "T2",
        "FLAIR",
    ]
    assert [trace.visible for trace in figure.data[:4]] == [False, False, False, True]
    assert [trace.name for trace in figure.data[4:]] == [
        "Tumor Core",
        "Edema",
        "Enhancing Tumor",
    ]
    assert len(figure.layout.updatemenus) == 3
    assert [frame.name for frame in figure.frames] == ["0", "2", "3", "4", "5"]


def test_nnunet_label_space_is_preserved_in_case(tmp_path: Path) -> None:
    """直接查看nnU-Net内部标签时应记录正确的标签空间。"""

    case_dir = tmp_path / "case"
    _write_visualization_case(case_dir)
    mask_path = case_dir / "BraTS2021_00001_seg.nii.gz"
    image = nib.load(mask_path)
    brats_mask = np.asarray(image.dataobj)
    nnunet_mask = np.zeros_like(brats_mask)
    nnunet_mask[brats_mask == 2] = 1
    nnunet_mask[brats_mask == 1] = 2
    nnunet_mask[brats_mask == 4] = 3
    nib.save(nib.Nifti1Image(nnunet_mask, image.affine), mask_path)

    case = load_visualization_case(
        case_dir,
        label_space=MaskLabelSpace.NNUNET,
    )

    assert case.label_space is MaskLabelSpace.NNUNET


def test_label_space_mismatch_is_rejected(tmp_path: Path) -> None:
    """避免把nnU-Net内部标签3误画成未知区域。"""

    case_dir = tmp_path / "case"
    _write_visualization_case(case_dir)
    mask_path = case_dir / "BraTS2021_00001_seg.nii.gz"
    image = nib.load(mask_path)
    mask = np.asarray(image.dataobj).copy()
    mask[mask == 4] = 3
    nib.save(nib.Nifti1Image(mask, image.affine), mask_path)

    with pytest.raises(VisualizationError, match="label-space"):
        load_visualization_case(case_dir, label_space=MaskLabelSpace.BRATS)


def test_mask_geometry_mismatch_is_rejected(tmp_path: Path) -> None:
    """mask尺寸与MRI不一致时禁止叠加。"""

    case_dir = tmp_path / "case"
    _write_visualization_case(case_dir, mask_shape=(7, 9, 6))

    with pytest.raises(VisualizationError, match="mask尺寸"):
        load_visualization_case(case_dir)


def test_case_without_mask_uses_middle_slice(tmp_path: Path) -> None:
    """不存在seg时仍可显示四模态，并默认选择中间层。"""

    case_dir = tmp_path / "case"
    _write_visualization_case(case_dir)
    (case_dir / "BraTS2021_00001_seg.nii.gz").unlink()

    case = load_visualization_case(case_dir)

    assert case.segmentation is None
    assert select_slice_index(case, SliceAxis.AXIAL) == 3
