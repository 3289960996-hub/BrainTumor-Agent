"""从 nnU-Net 分割掩膜提取适合 LLM Agent 使用的结构化 MRI 特征。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from data_process.io import read_nifti

LabelSpace = Literal["brats", "nnunet"]

SPATIAL_UNIT_TO_MM = {
    "unknown": 1.0,
    "mm": 1.0,
    "meter": 1000.0,
    "micron": 0.001,
}


class AnalysisError(ValueError):
    """分割结果无法被可靠分析时抛出的异常。"""


@dataclass(frozen=True, slots=True)
class TumorAnalysis:
    """提供给后续 RAG/LLM Agent 的稳定输出契约。"""

    tumor_volume: float
    location: str
    enhancing_ratio: float
    edema: bool

    def to_dict(self) -> dict[str, float | str | bool]:
        """转换为仅包含 JSON 基础类型的字典。"""

        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        """序列化为标准 UTF-8 JSON 文本。"""

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(frozen=True, slots=True)
class _RegionMasks:
    """统一后的 BraTS 派生区域。"""

    whole_tumor: NDArray[np.bool_]
    enhancing_tumor: NDArray[np.bool_]
    edema: NDArray[np.bool_]


def validate_and_round_labels(
    data: NDArray[np.floating],
) -> NDArray[np.int16]:
    """验证掩膜为有限整数标签，避免把概率图误当成分割结果。"""

    array = np.asarray(data)
    if not np.isfinite(array).all():
        raise AnalysisError("分割掩膜包含 NaN 或 Inf")

    rounded = np.rint(array)
    if not np.allclose(array, rounded, rtol=0.0, atol=1e-4):
        raise AnalysisError("输入包含非整数值；请传入离散分割 mask，而不是概率图")
    return rounded.astype(np.int16, copy=False)


def labels_to_regions(
    labels: NDArray[np.integer],
    label_space: LabelSpace = "brats",
) -> _RegionMasks:
    """把 BraTS 或 nnU-Net 标签转换成 WT、ET 和水肿区域。

    BraTS 标签为 0/1/2/4；本项目 nnU-Net 内部标签为 0/1/2/3。
    """

    values = np.asarray(labels, dtype=np.int16)
    unique_values = set(int(value) for value in np.unique(values))

    if label_space == "brats":
        unexpected = unique_values - {0, 1, 2, 4}
        if unexpected:
            raise AnalysisError(f"发现未知 BraTS 标签：{sorted(unexpected)}")
        return _RegionMasks(
            whole_tumor=np.isin(values, (1, 2, 4)),
            enhancing_tumor=values == 4,
            edema=values == 2,
        )

    if label_space == "nnunet":
        unexpected = unique_values - {0, 1, 2, 3}
        if unexpected:
            raise AnalysisError(f"发现未知 nnU-Net 标签：{sorted(unexpected)}")
        return _RegionMasks(
            whole_tumor=np.isin(values, (1, 2, 3)),
            enhancing_tumor=values == 3,
            edema=values == 1,
        )

    raise AnalysisError("label_space 必须是 brats 或 nnunet")


def spatial_unit_scale_to_mm(spatial_unit: str) -> float:
    """返回 NIfTI 空间单位到毫米的换算系数，unknown 按 BraTS 的毫米处理。"""

    if spatial_unit not in SPATIAL_UNIT_TO_MM:
        raise AnalysisError(f"不支持的 NIfTI 空间单位：{spatial_unit}")
    return SPATIAL_UNIT_TO_MM[spatial_unit]


def _voxel_volume_cm3(spacing: Sequence[float], spatial_unit: str) -> float:
    """根据 NIfTI spacing 和空间单位计算单体素体积（cm³）。"""

    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.shape != (3,) or np.any(spacing_array <= 0):
        raise AnalysisError(f"spacing 必须是三个正数，实际为：{tuple(spacing_array)}")

    spacing_mm = spacing_array * spatial_unit_scale_to_mm(spatial_unit)
    return float(np.prod(spacing_mm) / 1000.0)


def calculate_tumor_volume_cm3(
    whole_tumor: NDArray[np.bool_],
    spacing: Sequence[float],
    spatial_unit: str = "mm",
) -> float:
    """计算 Whole Tumor 体积，1 cm³ 等于 1000 mm³。"""

    voxel_count = int(np.count_nonzero(whole_tumor))
    return voxel_count * _voxel_volume_cm3(spacing, spatial_unit)


def calculate_enhancing_ratio(
    enhancing_tumor: NDArray[np.bool_],
    whole_tumor: NDArray[np.bool_],
) -> float:
    """计算增强肿瘤占 Whole Tumor 的体素比例。"""

    whole_count = int(np.count_nonzero(whole_tumor))
    if whole_count == 0:
        return 0.0
    return float(np.count_nonzero(enhancing_tumor) / whole_count)


def _canonical_tumor_mask(
    whole_tumor: NDArray[np.bool_],
    affine: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """把掩膜转换到最接近 RAS+ 的方向，消除存储方向对定位的影响。"""

    image = nib.Nifti1Image(
        np.asarray(whole_tumor, dtype=np.uint8),
        np.asarray(affine, dtype=np.float64),
    )
    canonical = nib.as_closest_canonical(image)
    return np.asarray(canonical.dataobj, dtype=np.uint8).astype(bool)


def infer_location_heuristic(
    whole_tumor: NDArray[np.bool_],
    affine: NDArray[np.float64],
) -> str:
    """依据 RAS+ 空间中的肿瘤质心给出粗粒度半球和脑叶位置。

    该方法不替代标准脑区 atlas。对于 BraTS 已配准、颅骨剥离数据可提供稳定的
    粗定位；需要精确脑叶定位时，应通过 ``atlas_path`` 使用配准后的分区图。
    """

    canonical_mask = _canonical_tumor_mask(whole_tumor, affine)
    coordinates = np.argwhere(canonical_mask)
    if coordinates.size == 0:
        return "none"

    shape = np.asarray(canonical_mask.shape, dtype=np.float64)
    normalized = (coordinates.astype(np.float64) + 0.5) / shape

    # RAS+ 第一个轴由左向右。跨中线且次要侧占比达到 25% 时报告 bilateral。
    left_count = int(np.count_nonzero(normalized[:, 0] < 0.5))
    right_count = int(coordinates.shape[0] - left_count)
    minor_share = min(left_count, right_count) / coordinates.shape[0]
    if left_count > 0 and right_count > 0 and minor_share >= 0.25:
        hemisphere = "bilateral"
    elif left_count >= right_count:
        hemisphere = "left"
    else:
        hemisphere = "right"

    centroid = normalized.mean(axis=0)
    anterior_position = float(centroid[1])
    superior_position = float(centroid[2])

    # RAS+ 第二轴由后向前，第三轴由下向上。
    if anterior_position >= 0.58:
        lobe = "frontal"
    elif anterior_position <= 0.30:
        lobe = "occipital"
    elif superior_position <= 0.42:
        lobe = "temporal"
    else:
        lobe = "parietal"
    return f"{hemisphere} {lobe}"


def load_atlas_label_map(path: str | Path) -> dict[int, str]:
    """读取 ``{"1": "left frontal"}`` 形式的 atlas 标签映射。"""

    label_map_path = Path(path).expanduser().resolve()
    if not label_map_path.is_file():
        raise AnalysisError(f"atlas 标签映射不存在：{label_map_path}")

    try:
        raw: Any = json.loads(label_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"无法读取 atlas 标签映射：{label_map_path}") from exc
    if not isinstance(raw, dict):
        raise AnalysisError("atlas 标签映射必须是 JSON object")

    parsed: dict[int, str] = {}
    for key, value in raw.items():
        try:
            label = int(key)
        except (TypeError, ValueError) as exc:
            raise AnalysisError(f"atlas 标签不是整数：{key!r}") from exc
        if label <= 0 or not isinstance(value, str) or not value.strip():
            raise AnalysisError(f"atlas 标签映射项无效：{key!r}: {value!r}")
        parsed[label] = value.strip()
    return parsed


def infer_location_from_atlas(
    whole_tumor: NDArray[np.bool_],
    mask_affine: NDArray[np.float64],
    atlas_path: str | Path,
    atlas_label_map: Mapping[int, str],
) -> str | None:
    """按 Whole Tumor 与配准 atlas 的最大重叠区域确定位置。"""

    atlas = read_nifti(atlas_path)
    if atlas.data.shape != whole_tumor.shape:
        raise AnalysisError(
            f"atlas shape={atlas.data.shape} 与 mask shape={whole_tumor.shape} 不一致"
        )
    if not np.allclose(atlas.geometry.affine, mask_affine, rtol=0.0, atol=1e-4):
        raise AnalysisError("atlas 与 mask 的 affine 不一致；请先使用最近邻插值完成配准")

    atlas_labels = validate_and_round_labels(atlas.data)
    overlapping = atlas_labels[np.asarray(whole_tumor, dtype=bool)]
    overlapping = overlapping[overlapping > 0]
    if overlapping.size == 0:
        return None

    labels, counts = np.unique(overlapping, return_counts=True)
    dominant_label = int(labels[int(np.argmax(counts))])
    if dominant_label not in atlas_label_map:
        raise AnalysisError(f"atlas 标签映射缺少重叠区域标签：{dominant_label}")
    return atlas_label_map[dominant_label]


def analyze_mask(
    mask_path: str | Path,
    label_space: LabelSpace = "brats",
    atlas_path: str | Path | None = None,
    atlas_label_map: Mapping[int, str] | None = None,
) -> TumorAnalysis:
    """分析一个 nnU-Net/BraTS NIfTI mask 并返回标准结果对象。"""

    if (atlas_path is None) != (atlas_label_map is None):
        raise AnalysisError("atlas_path 和 atlas_label_map 必须同时提供")

    volume = read_nifti(mask_path)
    labels = validate_and_round_labels(volume.data)
    regions = labels_to_regions(labels, label_space)

    spatial_unit = volume.header.get_xyzt_units()[0]
    location: str | None = None
    if atlas_path is not None and atlas_label_map is not None:
        location = infer_location_from_atlas(
            whole_tumor=regions.whole_tumor,
            mask_affine=volume.geometry.affine,
            atlas_path=atlas_path,
            atlas_label_map=atlas_label_map,
        )
    if location is None:
        location = infer_location_heuristic(
            whole_tumor=regions.whole_tumor,
            affine=volume.geometry.affine,
        )

    tumor_volume = calculate_tumor_volume_cm3(
        whole_tumor=regions.whole_tumor,
        spacing=volume.geometry.spacing,
        spatial_unit=spatial_unit,
    )
    enhancing_ratio = calculate_enhancing_ratio(
        enhancing_tumor=regions.enhancing_tumor,
        whole_tumor=regions.whole_tumor,
    )
    return TumorAnalysis(
        tumor_volume=round(tumor_volume, 3),
        location=location,
        enhancing_ratio=round(enhancing_ratio, 4),
        edema=bool(np.any(regions.edema)),
    )


def save_analysis(result: TumorAnalysis, output_path: str | Path) -> Path:
    """以 UTF-8 标准 JSON 保存分析结果。"""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.to_json(indent=2) + "\n", encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数。"""

    parser = argparse.ArgumentParser(
        description="从 nnU-Net 输出 mask 提取肿瘤体积、位置、增强比例和水肿状态。",
    )
    parser.add_argument("--mask", required=True, help="nnU-Net 输出的 .nii 或 .nii.gz mask")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSON；默认保存到 mask 同目录下的 *_analysis.json",
    )
    parser.add_argument(
        "--label-space",
        choices=["brats", "nnunet"],
        default="brats",
        help="brats=0/1/2/4；nnunet=本项目内部 0/1/2/3",
    )
    parser.add_argument("--atlas", default=None, help="可选：已配准到 mask 空间的脑区 atlas")
    parser.add_argument(
        "--atlas-label-map",
        default=None,
        help='可选：atlas JSON 标签映射，例如 {"1": "left frontal"}',
    )
    return parser


def _default_output_path(mask_path: str | Path) -> Path:
    """根据 NIfTI 文件名生成默认 JSON 路径。"""

    path = Path(mask_path).expanduser().resolve()
    name = path.name
    if name.lower().endswith(".nii.gz"):
        stem = name[:-7]
    elif name.lower().endswith(".nii"):
        stem = name[:-4]
    else:
        stem = path.stem
    return path.with_name(f"{stem}_analysis.json")


def main(argv: Sequence[str] | None = None) -> int:
    """执行单个分割 mask 的特征分析。"""

    args = build_parser().parse_args(argv)
    if (args.atlas is None) != (args.atlas_label_map is None):
        print("分析失败：--atlas 和 --atlas-label-map 必须同时提供")
        return 1

    try:
        atlas_mapping = (
            load_atlas_label_map(args.atlas_label_map)
            if args.atlas_label_map is not None
            else None
        )
        result = analyze_mask(
            mask_path=args.mask,
            label_space=args.label_space,
            atlas_path=args.atlas,
            atlas_label_map=atlas_mapping,
        )
        output_path = args.output or _default_output_path(args.mask)
        saved_path = save_analysis(result, output_path)
    except (AnalysisError, OSError, ValueError) as exc:
        print(f"分析失败：{exc}")
        return 1

    print(result.to_json(indent=2))
    print(f"JSON：{saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
