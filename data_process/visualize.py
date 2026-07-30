"""BraTS四模态MRI切片与肿瘤分割叠加可视化。

功能：
1. 使用Matplotlib生成T1、T1ce、T2、FLAIR四窗口静态切片；
2. 使用Plotly生成模拟放射科阅片的单窗口交互HTML；
3. 支持模态切换、切片滑块、播放和分割区域独立显隐；
4. 自动发现或显式加载BraTS/nnU-Net分割掩膜并进行透明叠加。

该模块只进行可视化，不包含预处理、分割推理或模型训练。
"""

import argparse
import base64
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray
from PIL import Image
from plotly.subplots import make_subplots

from data_process.constants import REQUIRED_MODALITIES, MRIModality
from data_process.exceptions import DataProcessError, VisualizationError
from data_process.io import is_nifti_path, read_nifti
from data_process.loader import load_brats_case
from data_process.schemas import LoadedStudy, NiftiVolume


class SliceAxis(StrEnum):
    """三维MRI的切片方向。"""

    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    AXIAL = "axial"


class MaskLabelSpace(StrEnum):
    """输入分割掩膜的标签空间。"""

    BRATS = "brats"
    NNUNET = "nnunet"


class TumorRegion(StrEnum):
    """阅片界面可独立控制的分割区域。"""

    TUMOR_CORE = "tumor_core"
    EDEMA = "edema"
    ENHANCING_TUMOR = "enhancing_tumor"


AXIS_INDEX: dict[SliceAxis, int] = {
    SliceAxis.SAGITTAL: 0,
    SliceAxis.CORONAL: 1,
    SliceAxis.AXIAL: 2,
}

MODALITY_TITLES: dict[MRIModality, str] = {
    MRIModality.T1: "T1",
    MRIModality.T1CE: "T1ce",
    MRIModality.T2: "T2",
    MRIModality.FLAIR: "FLAIR",
}

# 颜色按语义区域固定，避免在两种标签空间之间切换时改变医生看到的颜色。
REGION_COLORS: dict[TumorRegion, tuple[float, float, float]] = {
    TumorRegion.TUMOR_CORE: (0.95, 0.24, 0.24),
    TumorRegion.EDEMA: (0.98, 0.76, 0.18),
    TumorRegion.ENHANCING_TUMOR: (0.12, 0.86, 0.56),
}
REGION_TITLES: dict[TumorRegion, str] = {
    TumorRegion.TUMOR_CORE: "Tumor Core",
    TumorRegion.EDEMA: "Edema",
    TumorRegion.ENHANCING_TUMOR: "Enhancing Tumor",
}
REGION_LABELS: dict[MaskLabelSpace, dict[TumorRegion, tuple[int, ...]]] = {
    MaskLabelSpace.BRATS: {
        TumorRegion.TUMOR_CORE: (1, 4),
        TumorRegion.EDEMA: (2,),
        TumorRegion.ENHANCING_TUMOR: (4,),
    },
    MaskLabelSpace.NNUNET: {
        TumorRegion.TUMOR_CORE: (2, 3),
        TumorRegion.EDEMA: (1,),
        TumorRegion.ENHANCING_TUMOR: (3,),
    },
}

# 保留原有常量，供静态图和外部调用继续使用BraTS标签颜色。
MASK_COLORS: dict[int, tuple[float, float, float]] = {
    1: REGION_COLORS[TumorRegion.TUMOR_CORE],
    2: REGION_COLORS[TumorRegion.EDEMA],
    4: REGION_COLORS[TumorRegion.ENHANCING_TUMOR],
}
UNKNOWN_MASK_COLOR = (0.76, 0.32, 0.96)


@dataclass(frozen=True, slots=True)
class VisualizationCase:
    """已完成几何检查的四模态MRI及可选分割掩膜。"""

    study: LoadedStudy
    segmentation: NiftiVolume | None
    label_space: MaskLabelSpace = MaskLabelSpace.BRATS


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """可视化生成结果。"""

    case_id: str
    axis: SliceAxis
    slice_index: int
    static_path: Path
    interactive_path: Path | None
    segmentation_path: Path | None


def discover_segmentation(
    case_dir: str | Path,
    segmentation_path: str | Path | None = None,
) -> Path | None:
    """发现`*_seg.nii.gz`，显式路径优先。"""

    if segmentation_path is not None:
        explicit_path = Path(segmentation_path).expanduser().resolve()
        if not explicit_path.is_file() or not is_nifti_path(explicit_path):
            raise VisualizationError(f"分割掩膜不存在或不是NIfTI：{explicit_path}")
        return explicit_path

    directory = Path(case_dir).expanduser().resolve()
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and is_nifti_path(path)
        and (
            path.name.lower().endswith("_seg.nii.gz")
            or path.name.lower().endswith("_seg.nii")
        )
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise VisualizationError(f"发现多个seg mask，请使用--seg-path指定：{names}")
    return candidates[0]


def _validate_segmentation_geometry(
    reference: NiftiVolume,
    segmentation: NiftiVolume,
    label_space: MaskLabelSpace = MaskLabelSpace.BRATS,
    spacing_tolerance: float = 1e-5,
    affine_tolerance: float = 1e-4,
) -> None:
    """检查seg mask与MRI是否位于同一体素和物理空间。"""

    reference_geometry = reference.geometry
    mask_geometry = segmentation.geometry
    errors: list[str] = []

    if mask_geometry.shape != reference_geometry.shape:
        errors.append(
            f"mask尺寸{mask_geometry.shape}与MRI尺寸{reference_geometry.shape}不一致"
        )
    if not np.allclose(
        mask_geometry.spacing,
        reference_geometry.spacing,
        rtol=0.0,
        atol=spacing_tolerance,
    ):
        errors.append(
            f"mask spacing {mask_geometry.spacing}与MRI spacing "
            f"{reference_geometry.spacing}不一致"
        )
    if not np.allclose(
        mask_geometry.affine,
        reference_geometry.affine,
        rtol=0.0,
        atol=affine_tolerance,
    ):
        errors.append("mask affine与MRI affine不一致")

    rounded = np.rint(segmentation.data)
    if not np.allclose(segmentation.data, rounded, rtol=0.0, atol=1e-5):
        errors.append("seg mask包含非整数标签")
    else:
        allowed_labels = {
            0,
            *(
                label
                for region_labels in REGION_LABELS[label_space].values()
                for label in region_labels
            ),
        }
        present_labels = set(int(value) for value in np.unique(rounded))
        unexpected_labels = sorted(present_labels - allowed_labels)
        if unexpected_labels:
            errors.append(
                f"seg mask包含与{label_space.value}标签空间不兼容的值"
                f"{unexpected_labels}，请检查--label-space"
            )

    if errors:
        raise VisualizationError("；".join(errors))


def load_visualization_case(
    case_dir: str | Path,
    segmentation_path: str | Path | None = None,
    label_space: MaskLabelSpace = MaskLabelSpace.BRATS,
) -> VisualizationCase:
    """加载四模态MRI，并在seg存在时加载和校验掩膜。"""

    study = load_brats_case(case_dir)
    resolved_segmentation_path = discover_segmentation(case_dir, segmentation_path)
    segmentation = (
        read_nifti(resolved_segmentation_path)
        if resolved_segmentation_path is not None
        else None
    )
    if segmentation is not None:
        _validate_segmentation_geometry(
            reference=study.volumes[MRIModality.T1],
            segmentation=segmentation,
            label_space=label_space,
        )
    return VisualizationCase(
        study=study,
        segmentation=segmentation,
        label_space=label_space,
    )


def _extract_slice(
    volume: NDArray[np.floating],
    axis: SliceAxis,
    slice_index: int,
) -> NDArray[np.float32]:
    """从三维数组提取并旋转一个适合屏幕显示的二维切片。"""

    axis_index = AXIS_INDEX[axis]
    if slice_index < 0 or slice_index >= volume.shape[axis_index]:
        raise VisualizationError(
            f"{axis.value}切片索引{slice_index}越界，有效范围为"
            f"0至{volume.shape[axis_index] - 1}"
        )
    extracted = np.take(volume, indices=slice_index, axis=axis_index)
    return np.rot90(np.asarray(extracted, dtype=np.float32))


def select_slice_index(
    case: VisualizationCase,
    axis: SliceAxis = SliceAxis.AXIAL,
    requested_index: int | None = None,
) -> int:
    """选择切片：显式索引优先，否则选择肿瘤面积最大层或中间层。"""

    axis_length = case.study.reference_geometry.shape[AXIS_INDEX[axis]]
    if requested_index is not None:
        if requested_index < 0 or requested_index >= axis_length:
            raise VisualizationError(
                f"{axis.value}切片索引{requested_index}越界，有效范围为0至"
                f"{axis_length - 1}"
            )
        return requested_index

    if case.segmentation is None or not np.any(case.segmentation.data > 0):
        return axis_length // 2

    reduction_axes = tuple(index for index in range(3) if index != AXIS_INDEX[axis])
    tumor_area = np.count_nonzero(case.segmentation.data > 0, axis=reduction_axes)
    return int(np.argmax(tumor_area))


def _display_range(volume: NDArray[np.floating]) -> tuple[float, float]:
    """计算稳健显示窗，减少极端强度值对对比度的影响。"""

    finite = np.asarray(volume, dtype=np.float32)
    foreground = finite[finite != 0]
    values = foreground if foreground.size else finite.reshape(-1)
    lower, upper = np.percentile(values, (1.0, 99.0))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return float(lower), float(upper)


def _normalize_for_display(
    image_slice: NDArray[np.floating],
    value_range: tuple[float, float],
) -> NDArray[np.float32]:
    """将MRI显示强度压缩到0至1，不修改原始体素数据。"""

    lower, upper = value_range
    normalized = (np.asarray(image_slice, dtype=np.float32) - lower) / (upper - lower)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def _mask_rgba(
    mask_slice: NDArray[np.floating],
    alpha: float,
    label_space: MaskLabelSpace = MaskLabelSpace.BRATS,
) -> NDArray[np.float32]:
    """把BraTS或nnU-Net标签转换为透明RGBA叠加层。"""

    labels = np.rint(mask_slice).astype(np.int16)
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    known_labels: list[int] = []
    for region, color in REGION_COLORS.items():
        region_labels = REGION_LABELS[label_space][region]
        known_labels.extend(region_labels)
        selected = np.isin(labels, region_labels)
        rgba[selected, :3] = color
        rgba[selected, 3] = alpha

    unknown = (labels != 0) & ~np.isin(labels, known_labels)
    rgba[unknown, :3] = UNKNOWN_MASK_COLOR
    rgba[unknown, 3] = alpha
    return rgba


def _physical_aspect(study: LoadedStudy, axis: SliceAxis) -> float:
    """根据spacing计算Matplotlib二维像素纵横比。"""

    spacing_x, spacing_y, spacing_z = study.reference_geometry.spacing
    if axis is SliceAxis.AXIAL:
        return spacing_y / spacing_x
    if axis is SliceAxis.CORONAL:
        return spacing_z / spacing_x
    return spacing_z / spacing_y


def create_matplotlib_figure(
    case: VisualizationCase,
    axis: SliceAxis = SliceAxis.AXIAL,
    slice_index: int | None = None,
    mask_alpha: float = 0.45,
) -> tuple[Figure, int]:
    """创建2×2四模态Matplotlib静态切片。"""

    if not 0.0 <= mask_alpha <= 1.0:
        raise VisualizationError("mask_alpha必须位于0到1之间")

    selected_index = select_slice_index(case, axis, slice_index)
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), facecolor="#070b10")
    figure.suptitle(
        f"{case.study.case_id} · {axis.value.title()} slice {selected_index}",
        color="white",
        fontsize=16,
    )
    mask_slice = (
        _extract_slice(case.segmentation.data, axis, selected_index)
        if case.segmentation is not None
        else None
    )
    aspect = _physical_aspect(case.study, axis)

    for plot_axis, modality in zip(axes.flat, REQUIRED_MODALITIES, strict=True):
        volume = case.study.volumes[modality].data
        image_slice = _extract_slice(volume, axis, selected_index)
        display_image = _normalize_for_display(image_slice, _display_range(volume))

        plot_axis.imshow(
            display_image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect=aspect,
        )
        if mask_slice is not None:
            plot_axis.imshow(
                _mask_rgba(mask_slice, mask_alpha, case.label_space),
                interpolation="nearest",
                aspect=aspect,
            )
        plot_axis.set_title(MODALITY_TITLES[modality], color="white", fontsize=14)
        plot_axis.set_axis_off()

    if mask_slice is not None:
        legend_items = [
            Patch(
                color=REGION_COLORS[region],
                label=(
                    f"{REGION_TITLES[region]} "
                    f"({'+'.join(str(label) for label in REGION_LABELS[case.label_space][region])})"
                ),
            )
            for region in TumorRegion
        ]
        figure.legend(
            handles=legend_items,
            loc="lower center",
            ncol=3,
            frameon=False,
            labelcolor="white",
        )

    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    return figure, selected_index


def _composited_png_source(
    image_slice: NDArray[np.floating],
    value_range: tuple[float, float],
    mask_slice: NDArray[np.floating] | None,
    mask_alpha: float,
    label_space: MaskLabelSpace = MaskLabelSpace.BRATS,
) -> str:
    """把MRI和mask合成为PNG Data URL，降低Plotly HTML体积。"""

    display_image = _normalize_for_display(image_slice, value_range)
    rgb = np.repeat(display_image[..., np.newaxis], repeats=3, axis=2)
    if mask_slice is not None:
        overlay = _mask_rgba(mask_slice, mask_alpha, label_space)
        overlay_alpha = overlay[..., 3:4]
        rgb = rgb * (1.0 - overlay_alpha) + overlay[..., :3] * overlay_alpha

    uint8_image = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(uint8_image).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mri_png_source(
    image_slice: NDArray[np.floating],
    value_range: tuple[float, float],
) -> str:
    """将单张MRI切片编码为灰阶RGB PNG Data URL。"""

    display_image = _normalize_for_display(image_slice, value_range)
    grayscale = np.rint(display_image * 255.0).astype(np.uint8)
    rgb = np.repeat(grayscale[..., np.newaxis], repeats=3, axis=2)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _region_overlay_png_source(
    mask_slice: NDArray[np.floating],
    region: TumorRegion,
    label_space: MaskLabelSpace,
    alpha: float,
) -> str:
    """把一个肿瘤语义区域编码为带透明通道的PNG Data URL。"""

    labels = np.rint(mask_slice).astype(np.int16)
    selected = np.isin(labels, REGION_LABELS[label_space][region])
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    color = np.rint(np.asarray(REGION_COLORS[region]) * 255.0).astype(np.uint8)
    rgba[selected, :3] = color
    rgba[selected, 3] = int(round(alpha * 255.0))

    buffer = BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _viewer_slice_indices(
    axis_length: int,
    selected_index: int,
    slice_step: int,
) -> list[int]:
    """生成滑块使用的切片序列，并确保包含末层和初始层。"""

    if slice_step < 1:
        raise VisualizationError("slice_step必须大于或等于1")
    indices = list(range(0, axis_length, slice_step))
    if axis_length - 1 not in indices:
        indices.append(axis_length - 1)
    if selected_index not in indices:
        indices.append(selected_index)
        indices.sort()
    return indices


def _region_legend_html() -> str:
    """生成阅片界面使用的彩色区域图例。"""

    items: list[str] = []
    for region in TumorRegion:
        red, green, blue = (
            int(round(channel * 255.0)) for channel in REGION_COLORS[region]
        )
        items.append(
            f"<span style='color:rgb({red},{green},{blue})'>■</span> "
            f"{REGION_TITLES[region]}"
        )
    return "&nbsp;&nbsp;&nbsp;".join(items)


def create_radiology_viewer_figure(
    case: VisualizationCase,
    axis: SliceAxis = SliceAxis.AXIAL,
    initial_slice_index: int | None = None,
    initial_modality: MRIModality = MRIModality.FLAIR,
    mask_alpha: float = 0.45,
    slice_step: int = 1,
) -> go.Figure:
    """创建模拟放射科阅片器的Plotly单窗口交互图。

    四个MRI模态作为互斥底图，三个肿瘤语义区域作为独立透明图层。医生可以
    切换模态、滚动/播放切片，并分别隐藏或显示Core、Edema和Enhancing区域。
    """

    if not 0.0 <= mask_alpha <= 1.0:
        raise VisualizationError("mask_alpha必须位于0到1之间")
    if initial_modality not in REQUIRED_MODALITIES:
        raise VisualizationError(f"不支持的初始模态：{initial_modality}")

    selected_index = select_slice_index(case, axis, initial_slice_index)
    axis_length = case.study.reference_geometry.shape[AXIS_INDEX[axis]]
    slice_indices = _viewer_slice_indices(axis_length, selected_index, slice_step)
    value_ranges = {
        modality: _display_range(case.study.volumes[modality].data)
        for modality in REQUIRED_MODALITIES
    }

    def sources_for_slice(index: int) -> list[str]:
        sources = [
            _mri_png_source(
                image_slice=_extract_slice(
                    case.study.volumes[modality].data,
                    axis,
                    index,
                ),
                value_range=value_ranges[modality],
            )
            for modality in REQUIRED_MODALITIES
        ]
        if case.segmentation is None:
            return sources

        mask_slice = _extract_slice(case.segmentation.data, axis, index)
        sources.extend(
            _region_overlay_png_source(
                mask_slice=mask_slice,
                region=region,
                label_space=case.label_space,
                alpha=mask_alpha,
            )
            for region in TumorRegion
        )
        return sources

    initial_sources = sources_for_slice(selected_index)
    initial_modality_index = REQUIRED_MODALITIES.index(initial_modality)
    figure = go.Figure()

    for index, (source, modality) in enumerate(
        zip(initial_sources[: len(REQUIRED_MODALITIES)], REQUIRED_MODALITIES, strict=True)
    ):
        figure.add_trace(
            go.Image(
                source=source,
                name=MODALITY_TITLES[modality],
                visible=index == initial_modality_index,
                hoverinfo="skip",
            )
        )

    if case.segmentation is not None:
        for source, region in zip(
            initial_sources[len(REQUIRED_MODALITIES) :],
            TumorRegion,
            strict=True,
        ):
            figure.add_trace(
                go.Image(
                    source=source,
                    name=REGION_TITLES[region],
                    visible=True,
                    hoverinfo="skip",
                )
            )

    trace_indices = list(range(len(initial_sources)))
    frames: list[go.Frame] = []
    for index in slice_indices:
        sources = initial_sources if index == selected_index else sources_for_slice(index)
        frames.append(
            go.Frame(
                name=str(index),
                data=[go.Image(source=source) for source in sources],
                traces=trace_indices,
            )
        )
    figure.frames = frames

    modality_buttons = [
        {
            "label": MODALITY_TITLES[modality],
            "method": "restyle",
            "args": [
                {
                    "visible": [
                        candidate is modality for candidate in REQUIRED_MODALITIES
                    ]
                },
                list(range(len(REQUIRED_MODALITIES))),
            ],
        }
        for modality in REQUIRED_MODALITIES
    ]
    update_menus: list[dict] = [
        {
            "type": "dropdown",
            "x": 0.0,
            "y": 1.13,
            "xanchor": "left",
            "yanchor": "top",
            "showactive": True,
            "active": initial_modality_index,
            "buttons": modality_buttons,
        }
    ]

    if case.segmentation is not None:
        overlay_trace_indices = list(
            range(
                len(REQUIRED_MODALITIES),
                len(REQUIRED_MODALITIES) + len(TumorRegion),
            )
        )
        overlay_buttons = [
            ("All regions", [True, True, True]),
            ("Tumor Core", [True, False, False]),
            ("Edema", [False, True, False]),
            ("Enhancing", [False, False, True]),
            ("MRI only", [False, False, False]),
        ]
        update_menus.append(
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.24,
                "y": 1.13,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True,
                "buttons": [
                    {
                        "label": label,
                        "method": "restyle",
                        "args": [{"visible": visible}, overlay_trace_indices],
                    }
                    for label, visible in overlay_buttons
                ],
            }
        )

    update_menus.append(
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.0,
            "y": -0.08,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "fromcurrent": True,
                            "frame": {"duration": 120, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "Ⅱ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    )

    slider_steps = [
        {
            "method": "animate",
            "label": str(index),
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for index in slice_indices
    ]
    annotations = []
    if case.segmentation is not None:
        annotations.append(
            {
                "text": _region_legend_html(),
                "x": 0.5,
                "y": -0.18,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 13, "color": "#d7e0ea"},
            }
        )

    figure.update_layout(
        title={
            "text": (
                f"{case.study.case_id} · AI Segmentation Review · "
                f"{axis.value.title()}"
            ),
            "x": 0.5,
        },
        template="plotly_dark",
        paper_bgcolor="#070b10",
        plot_bgcolor="#070b10",
        height=850,
        margin={"l": 30, "r": 30, "t": 145, "b": 150},
        annotations=annotations,
        updatemenus=update_menus,
        sliders=[
            {
                "active": slice_indices.index(selected_index),
                "currentvalue": {"prefix": f"{axis.value.title()} slice: "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
    )
    figure.update_xaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        constrain="domain",
    )
    figure.update_yaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
    )
    return figure


def create_plotly_figure(
    case: VisualizationCase,
    axis: SliceAxis = SliceAxis.AXIAL,
    initial_slice_index: int | None = None,
    mask_alpha: float = 0.45,
    slice_step: int = 1,
) -> go.Figure:
    """创建带滑块、播放和暂停控制的四模态Plotly切片图。"""

    if slice_step < 1:
        raise VisualizationError("slice_step必须大于或等于1")
    if not 0.0 <= mask_alpha <= 1.0:
        raise VisualizationError("mask_alpha必须位于0到1之间")

    selected_index = select_slice_index(case, axis, initial_slice_index)
    axis_length = case.study.reference_geometry.shape[AXIS_INDEX[axis]]
    slice_indices = list(range(0, axis_length, slice_step))
    if axis_length - 1 not in slice_indices:
        slice_indices.append(axis_length - 1)
    if selected_index not in slice_indices:
        slice_indices.append(selected_index)
        slice_indices.sort()

    value_ranges = {
        modality: _display_range(case.study.volumes[modality].data)
        for modality in REQUIRED_MODALITIES
    }

    def sources_for_slice(index: int) -> list[str]:
        mask_slice = (
            _extract_slice(case.segmentation.data, axis, index)
            if case.segmentation is not None
            else None
        )
        return [
            _composited_png_source(
                image_slice=_extract_slice(
                    case.study.volumes[modality].data,
                    axis,
                    index,
                ),
                value_range=value_ranges[modality],
                mask_slice=mask_slice,
                mask_alpha=mask_alpha,
                label_space=case.label_space,
            )
            for modality in REQUIRED_MODALITIES
        ]

    initial_sources = sources_for_slice(selected_index)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[MODALITY_TITLES[item] for item in REQUIRED_MODALITIES],
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )
    positions = ((1, 1), (1, 2), (2, 1), (2, 2))
    for source, (row, column), modality in zip(
        initial_sources,
        positions,
        REQUIRED_MODALITIES,
        strict=True,
    ):
        figure.add_trace(
            go.Image(source=source, name=MODALITY_TITLES[modality], hoverinfo="skip"),
            row=row,
            col=column,
        )

    frames: list[go.Frame] = []
    for index in slice_indices:
        sources = initial_sources if index == selected_index else sources_for_slice(index)
        frames.append(
            go.Frame(
                name=str(index),
                data=[go.Image(source=source) for source in sources],
                traces=[0, 1, 2, 3],
            )
        )
    figure.frames = frames

    slider_steps = [
        {
            "method": "animate",
            "label": str(index),
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for index in slice_indices
    ]
    figure.update_layout(
        title=f"{case.study.case_id} · {axis.value.title()} MRI",
        template="plotly_dark",
        paper_bgcolor="#070b10",
        plot_bgcolor="#070b10",
        margin={"l": 20, "r": 20, "t": 80, "b": 120},
        sliders=[
            {
                "active": slice_indices.index(selected_index),
                "currentvalue": {"prefix": "Slice: "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 120, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
    )
    figure.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    figure.update_yaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        scaleanchor=None,
    )
    for annotation in figure.layout.annotations:
        annotation.font.color = "white"
        annotation.font.size = 14
    return figure


def generate_visualizations(
    case_dir: str | Path,
    output_dir: str | Path,
    segmentation_path: str | Path | None = None,
    axis: SliceAxis = SliceAxis.AXIAL,
    slice_index: int | None = None,
    mask_alpha: float = 0.45,
    slice_step: int = 1,
    create_interactive: bool = True,
    show: bool = False,
    label_space: MaskLabelSpace = MaskLabelSpace.BRATS,
    initial_modality: MRIModality = MRIModality.FLAIR,
) -> VisualizationResult:
    """加载病例并生成四窗Matplotlib PNG与交互式阅片HTML。"""

    case = load_visualization_case(
        case_dir,
        segmentation_path,
        label_space=label_space,
    )
    target_dir = Path(output_dir).expanduser().resolve() / case.study.case_id
    target_dir.mkdir(parents=True, exist_ok=True)

    static_figure, selected_index = create_matplotlib_figure(
        case=case,
        axis=axis,
        slice_index=slice_index,
        mask_alpha=mask_alpha,
    )
    static_path = target_dir / (
        f"{case.study.case_id}_{axis.value}_slice_{selected_index}.png"
    )
    static_figure.savefig(
        static_path,
        dpi=160,
        bbox_inches="tight",
        facecolor=static_figure.get_facecolor(),
    )
    if show:
        plt.show()
    plt.close(static_figure)

    interactive_path: Path | None = None
    if create_interactive:
        plotly_figure = create_radiology_viewer_figure(
            case=case,
            axis=axis,
            initial_slice_index=selected_index,
            initial_modality=initial_modality,
            mask_alpha=mask_alpha,
            slice_step=slice_step,
        )
        interactive_path = target_dir / (
            f"{case.study.case_id}_{axis.value}_radiology_viewer.html"
        )
        plotly_figure.write_html(
            interactive_path,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
            },
        )

    return VisualizationResult(
        case_id=case.study.case_id,
        axis=axis,
        slice_index=selected_index,
        static_path=static_path,
        interactive_path=interactive_path,
        segmentation_path=case.segmentation.path if case.segmentation is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    """创建MRI可视化命令行参数。"""

    parser = argparse.ArgumentParser(
        description="生成BraTS四模态MRI静态切片和交互式Plotly HTML。",
    )
    parser.add_argument("--input-dir", required=True, help="BraTS单病例目录")
    parser.add_argument(
        "--output-dir",
        default="./runtime/visualizations",
        help="可视化输出根目录，默认./runtime/visualizations",
    )
    parser.add_argument(
        "--seg-path",
        default=None,
        help="可选seg mask路径；未指定时自动发现*_seg.nii.gz",
    )
    parser.add_argument(
        "--axis",
        choices=[axis.value for axis in SliceAxis],
        default=SliceAxis.AXIAL.value,
        help="切片方向，默认axial",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="静态切片索引；未指定时选择肿瘤面积最大层或中间层",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="mask透明度，范围0至1，默认0.45",
    )
    parser.add_argument(
        "--label-space",
        choices=[space.value for space in MaskLabelSpace],
        default=MaskLabelSpace.BRATS.value,
        help="mask标签空间：brats=0/1/2/4，nnunet=0/1/2/3",
    )
    parser.add_argument(
        "--initial-modality",
        choices=[modality.value for modality in REQUIRED_MODALITIES],
        default=MRIModality.FLAIR.value,
        help="交互阅片器初始显示模态，默认flair",
    )
    parser.add_argument(
        "--slice-step",
        type=int,
        default=1,
        help="Plotly滑块采样步长，默认包含每一层",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="只生成Matplotlib PNG，不生成Plotly HTML",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="生成后打开Matplotlib窗口；服务器环境不建议使用",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行MRI可视化命令。"""

    args = build_parser().parse_args(argv)
    try:
        result = generate_visualizations(
            case_dir=args.input_dir,
            output_dir=args.output_dir,
            segmentation_path=args.seg_path,
            axis=SliceAxis(args.axis),
            slice_index=args.slice_index,
            mask_alpha=args.mask_alpha,
            slice_step=args.slice_step,
            create_interactive=not args.no_html,
            show=args.show,
            label_space=MaskLabelSpace(args.label_space),
            initial_modality=MRIModality(args.initial_modality),
        )
    except DataProcessError as exc:
        print(f"可视化失败：{exc}")
        return 1

    print(f"可视化完成：{result.case_id}")
    print(f"  静态图：{result.static_path}")
    if result.interactive_path is not None:
        print(f"  交互图：{result.interactive_path}")
    if result.segmentation_path is None:
        print("  未发现seg mask，本次仅显示MRI")
    else:
        print(f"  seg mask：{result.segmentation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
