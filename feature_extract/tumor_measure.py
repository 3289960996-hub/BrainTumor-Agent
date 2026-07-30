"""BraTS/nnU-Net 脑肿瘤分割 mask 的定量测量模块。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

from data_process.io import read_nifti
from feature_extract.analyzer import (
    AnalysisError,
    LabelSpace,
    calculate_tumor_volume_cm3,
    infer_location_from_atlas,
    infer_location_heuristic,
    load_atlas_label_map,
    spatial_unit_scale_to_mm,
    validate_and_round_labels,
)


@dataclass(frozen=True, slots=True)
class TumorRegions:
    """BraTS 临床区域的布尔掩膜。"""

    whole_tumor: NDArray[np.bool_]
    tumor_core: NDArray[np.bool_]
    enhancing_tumor: NDArray[np.bool_]
    edema: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class TumorMeasurement:
    """适合直接传递给 LLM Agent 的量化结果。

    ``tumor_volume`` 即 Whole Tumor 体积。所有 ratio 均以 Whole Tumor
    体素数为分母；无肿瘤时比例定义为 0。
    """

    tumor_volume: float
    tumor_core_volume: float
    enhancing_volume: float
    max_diameter: float
    edema: bool
    location: str
    edema_volume: float
    tumor_core_ratio: float
    enhancing_ratio: float
    edema_ratio: float

    def to_dict(self) -> dict[str, float | str | bool]:
        """转换为只包含 JSON 基础类型的 Agent payload。"""

        return asdict(self)

    def to_summary_dict(self) -> dict[str, float | str | bool]:
        """返回与基础六字段接口完全一致的精简结果。"""

        return {
            "tumor_volume": self.tumor_volume,
            "tumor_core_volume": self.tumor_core_volume,
            "enhancing_volume": self.enhancing_volume,
            "max_diameter": self.max_diameter,
            "edema": self.edema,
            "location": self.location,
        }

    def to_json(self, indent: int | None = 2, include_ratios: bool = True) -> str:
        """序列化为标准 JSON；默认附带水肿体积和区域占比。"""

        payload = self.to_dict() if include_ratios else self.to_summary_dict()
        return json.dumps(payload, ensure_ascii=False, indent=indent)


def labels_to_tumor_regions(
    labels: NDArray[np.integer],
    label_space: LabelSpace = "brats",
) -> TumorRegions:
    """将离散标签转换为 WT、TC、ET 和水肿区域。"""

    values = np.asarray(labels, dtype=np.int16)
    unique_values = set(int(value) for value in np.unique(values))

    if label_space == "brats":
        unexpected = unique_values - {0, 1, 2, 4}
        if unexpected:
            raise AnalysisError(f"发现未知 BraTS 标签：{sorted(unexpected)}")
        return TumorRegions(
            whole_tumor=np.isin(values, (1, 2, 4)),
            tumor_core=np.isin(values, (1, 4)),
            enhancing_tumor=values == 4,
            edema=values == 2,
        )

    if label_space == "nnunet":
        unexpected = unique_values - {0, 1, 2, 3}
        if unexpected:
            raise AnalysisError(f"发现未知 nnU-Net 标签：{sorted(unexpected)}")
        return TumorRegions(
            whole_tumor=np.isin(values, (1, 2, 3)),
            tumor_core=np.isin(values, (2, 3)),
            enhancing_tumor=values == 3,
            edema=values == 1,
        )

    raise AnalysisError("label_space 必须是 brats 或 nnunet")


def calculate_region_ratio(
    region: NDArray[np.bool_],
    whole_tumor: NDArray[np.bool_],
) -> float:
    """计算指定区域占 Whole Tumor 的体素比例。"""

    whole_count = int(np.count_nonzero(whole_tumor))
    if whole_count == 0:
        return 0.0
    return float(np.count_nonzero(region) / whole_count)


def _boundary_physical_points(
    whole_tumor: NDArray[np.bool_],
    affine: NDArray[np.float64],
    spatial_unit: str,
) -> NDArray[np.float64]:
    """提取 WT 边界体素中心，并转换到毫米物理空间。"""

    mask = np.asarray(whole_tumor, dtype=bool)
    if mask.ndim != 3:
        raise AnalysisError(f"Whole Tumor mask 必须是三维，实际维度为 {mask.ndim}")
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float64)

    structure = generate_binary_structure(rank=3, connectivity=1)
    boundary = mask & ~binary_erosion(mask, structure=structure, border_value=0)
    voxel_coordinates = np.argwhere(boundary)
    physical_points = nib.affines.apply_affine(
        np.asarray(affine, dtype=np.float64),
        voxel_coordinates,
    )
    return np.asarray(physical_points, dtype=np.float64) * spatial_unit_scale_to_mm(
        spatial_unit
    )


def _convex_hull_vertices(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """用内在维度凸包减少候选点，并处理平面或直线病灶。"""

    if len(points) < 3:
        return points

    centered = points - points.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        return points[:1]
    if rank == 1:
        positions = centered @ right_vectors[0]
        endpoint_indices = np.unique([int(np.argmin(positions)), int(np.argmax(positions))])
        return points[endpoint_indices]

    projected = centered @ right_vectors[:rank].T
    try:
        hull = ConvexHull(projected)
    except QhullError:
        return points
    return points[hull.vertices]


def _maximum_pairwise_distance(
    points: NDArray[np.float64],
    block_size: int = 2048,
) -> float:
    """分块计算最大点间距，避免为大型病灶分配完整距离矩阵。"""

    point_count = len(points)
    if point_count < 2:
        return 0.0
    if block_size < 1:
        raise AnalysisError("block_size 必须大于或等于 1")

    maximum = 0.0
    for start in range(0, point_count, block_size):
        block = points[start : start + block_size]
        maximum = max(maximum, float(np.max(cdist(block, points, metric="euclidean"))))
    return maximum


def calculate_max_diameter_mm(
    whole_tumor: NDArray[np.bool_],
    affine: NDArray[np.float64],
    spatial_unit: str = "mm",
    block_size: int = 2048,
) -> float:
    """计算 WT 边界体素中心之间的最大三维欧氏距离（毫米）。

    先提取 6 邻域边界，再使用凸包减少候选点，最后分块求精确最大点间距。
    """

    boundary_points = _boundary_physical_points(
        whole_tumor=whole_tumor,
        affine=affine,
        spatial_unit=spatial_unit,
    )
    hull_points = _convex_hull_vertices(boundary_points)
    return _maximum_pairwise_distance(hull_points, block_size=block_size)


def measure_tumor(
    mask_path: str | Path,
    label_space: LabelSpace = "brats",
    atlas_path: str | Path | None = None,
    atlas_label_map: Mapping[int, str] | None = None,
) -> TumorMeasurement:
    """读取单个 segmentation mask 并完成全部定量分析。"""

    if (atlas_path is None) != (atlas_label_map is None):
        raise AnalysisError("atlas_path 和 atlas_label_map 必须同时提供")

    volume = read_nifti(mask_path)
    labels = validate_and_round_labels(volume.data)
    regions = labels_to_tumor_regions(labels, label_space=label_space)
    spatial_unit = volume.header.get_xyzt_units()[0]
    spacing = volume.geometry.spacing

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
        regions.whole_tumor,
        spacing,
        spatial_unit,
    )
    tumor_core_volume = calculate_tumor_volume_cm3(
        regions.tumor_core,
        spacing,
        spatial_unit,
    )
    enhancing_volume = calculate_tumor_volume_cm3(
        regions.enhancing_tumor,
        spacing,
        spatial_unit,
    )
    edema_volume = calculate_tumor_volume_cm3(
        regions.edema,
        spacing,
        spatial_unit,
    )

    return TumorMeasurement(
        tumor_volume=round(tumor_volume, 3),
        tumor_core_volume=round(tumor_core_volume, 3),
        enhancing_volume=round(enhancing_volume, 3),
        max_diameter=round(
            calculate_max_diameter_mm(
                whole_tumor=regions.whole_tumor,
                affine=volume.geometry.affine,
                spatial_unit=spatial_unit,
            ),
            2,
        ),
        edema=bool(np.any(regions.edema)),
        location=location,
        edema_volume=round(edema_volume, 3),
        tumor_core_ratio=round(
            calculate_region_ratio(regions.tumor_core, regions.whole_tumor),
            4,
        ),
        enhancing_ratio=round(
            calculate_region_ratio(regions.enhancing_tumor, regions.whole_tumor),
            4,
        ),
        edema_ratio=round(
            calculate_region_ratio(regions.edema, regions.whole_tumor),
            4,
        ),
    )


def save_measurement(
    result: TumorMeasurement,
    output_path: str | Path,
    include_ratios: bool = True,
) -> Path:
    """保存 UTF-8 JSON，默认包含量化比例。"""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        result.to_json(indent=2, include_ratios=include_ratios) + "\n",
        encoding="utf-8",
    )
    return target


def _default_output_path(mask_path: str | Path) -> Path:
    """根据 NIfTI 文件名生成默认结果路径。"""

    path = Path(mask_path).expanduser().resolve()
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        stem = path.name[:-7]
    elif lower_name.endswith(".nii"):
        stem = path.name[:-4]
    else:
        stem = path.stem
    return path.with_name(f"{stem}_measurement.json")


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数。"""

    parser = argparse.ArgumentParser(
        description="量化 nnU-Net/BraTS 脑肿瘤分割 mask。",
    )
    parser.add_argument("--mask", required=True, help="输入 .nii 或 .nii.gz 分割 mask")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument(
        "--label-space",
        choices=["brats", "nnunet"],
        default="brats",
        help="brats=0/1/2/4；nnunet=本项目内部 0/1/2/3",
    )
    parser.add_argument("--atlas", default=None, help="可选：已配准到 mask 空间的脑区 atlas")
    parser.add_argument("--atlas-label-map", default=None, help="可选：atlas 标签映射 JSON")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="仅输出用户约定的六个基础字段，不附带区域体积和比例",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行单个 mask 的量化分析并保存 JSON。"""

    args = build_parser().parse_args(argv)
    if (args.atlas is None) != (args.atlas_label_map is None):
        print("测量失败：--atlas 和 --atlas-label-map 必须同时提供")
        return 1

    try:
        atlas_mapping = (
            load_atlas_label_map(args.atlas_label_map)
            if args.atlas_label_map is not None
            else None
        )
        result = measure_tumor(
            mask_path=args.mask,
            label_space=args.label_space,
            atlas_path=args.atlas,
            atlas_label_map=atlas_mapping,
        )
        output_path = args.output or _default_output_path(args.mask)
        saved_path = save_measurement(
            result,
            output_path,
            include_ratios=not args.summary_only,
        )
    except (AnalysisError, OSError, ValueError) as exc:
        print(f"测量失败：{exc}")
        return 1

    print(result.to_json(indent=2, include_ratios=not args.summary_only))
    print(f"JSON：{saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
