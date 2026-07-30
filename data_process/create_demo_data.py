"""生成用于BrainTumor-Agent工程联调的轻量四模态NIfTI病例。

说明：
1. 基础解剖数据来自NiBabel随包测试数据；
2. 肿瘤及模态信号为程序合成，不代表真实患者或真实影像表现；
3. 输出适合测试上传、NIfTI解析、阅片、Mask叠加和接口流程；
4. 不得用于模型效果评价、医学研究结论或临床用途。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.testing import data_path
from scipy.ndimage import gaussian_filter, zoom

TARGET_SHAPE = (96, 112, 80)
AFFINE = np.diag([1.5, 1.5, 1.5, 1.0])


def _load_base_volume() -> np.ndarray:
    """加载并重采样NiBabel自带的三维解剖测试影像。"""

    source_path = Path(data_path) / "anatomical.nii"
    source = np.asarray(nib.load(source_path).get_fdata(), dtype=np.float32)
    factors = tuple(
        target / current
        for target, current in zip(TARGET_SHAPE, source.shape, strict=True)
    )
    resized = zoom(source, factors, order=3)

    finite = resized[np.isfinite(resized)]
    low, high = np.percentile(finite, (2.0, 99.5))
    normalized = np.clip((resized - low) / max(high - low, 1e-6), 0.0, 1.0)
    normalized = gaussian_filter(normalized, sigma=0.55)
    normalized[normalized < 0.035] = 0.0
    return normalized.astype(np.float32)


def _ellipsoid(
    center: tuple[float, float, float],
    radius: tuple[float, float, float],
) -> np.ndarray:
    """返回指定中心和半径的三维椭球布尔Mask。"""

    coordinates = np.indices(TARGET_SHAPE, dtype=np.float32)
    distance = sum(
        ((coordinates[axis] - center[axis]) / radius[axis]) ** 2
        for axis in range(3)
    )
    return distance <= 1.0


def _build_case(
    base: np.ndarray,
    *,
    case_id: str,
    center: tuple[float, float, float],
    radius: tuple[float, float, float],
    seed: int,
) -> dict[str, np.ndarray]:
    """构造四模态信号及BraTS标签空间的合成分割Mask。"""

    rng = np.random.default_rng(seed)
    brain = base > 0.035
    edema = _ellipsoid(center, radius) & brain
    core_radius = tuple(value * 0.58 for value in radius)
    core = _ellipsoid(center, core_radius) & brain
    enhancing_outer = _ellipsoid(
        center,
        tuple(value * 0.56 for value in radius),
    )
    enhancing_inner = _ellipsoid(
        center,
        tuple(value * 0.30 for value in radius),
    )
    enhancing = enhancing_outer & ~enhancing_inner & brain
    necrosis = enhancing_inner & brain

    seg = np.zeros(TARGET_SHAPE, dtype=np.uint8)
    seg[edema] = 2
    seg[core] = 1
    seg[enhancing] = 4
    seg[necrosis] = 1

    noise = rng.normal(0.0, 0.018, TARGET_SHAPE).astype(np.float32)
    tissue = np.clip(base + noise, 0.0, 1.0) * brain

    t1 = tissue * 900.0
    t1[edema] *= 0.86
    t1[core] *= 0.72

    t1ce = tissue * 920.0
    t1ce[edema] *= 0.90
    t1ce[core] *= 0.70
    t1ce[enhancing] += 620.0
    t1ce[necrosis] *= 0.48

    t2 = (0.68 * tissue + 0.20 * (1.0 - tissue)) * 820.0 * brain
    t2[edema] += 420.0
    t2[core] += 170.0

    flair = tissue * 720.0
    flair[edema] += 560.0
    flair[core] += 230.0
    flair[necrosis] *= 0.62

    return {
        f"{case_id}_t1.nii.gz": t1.astype(np.float32),
        f"{case_id}_t1ce.nii.gz": t1ce.astype(np.float32),
        f"{case_id}_t2.nii.gz": t2.astype(np.float32),
        f"{case_id}_flair.nii.gz": flair.astype(np.float32),
        f"{case_id}_seg.nii.gz": seg,
    }


def _save_case(
    output_root: Path,
    case_id: str,
    arrays: dict[str, np.ndarray],
) -> Path:
    """保存病例并生成便于传输的ZIP压缩包。"""

    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for filename, array in arrays.items():
        image = nib.Nifti1Image(array, AFFINE)
        image.set_qform(AFFINE, code=1)
        image.set_sform(AFFINE, code=1)
        nib.save(image, case_dir / filename)

    archive_path = output_root / f"{case_id}.zip"
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(
        str(output_root / case_id),
        "zip",
        root_dir=case_dir,
    )
    return case_dir


def create_demo_dataset(output_root: str | Path) -> list[Path]:
    """生成两个位置和大小不同的工程联调病例。"""

    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    base = _load_base_volume()
    specifications = [
        (
            "BT-Demo-001",
            (62.0, 60.0, 42.0),
            (15.0, 18.0, 13.0),
            202101,
        ),
        (
            "BT-Demo-002",
            (34.0, 52.0, 36.0),
            (12.0, 15.0, 11.0),
            202102,
        ),
    ]

    created = []
    for case_id, center, radius, seed in specifications:
        arrays = _build_case(
            base,
            case_id=case_id,
            center=center,
            radius=radius,
            seed=seed,
        )
        created.append(_save_case(destination, case_id, arrays))

    readme = destination / "使用说明.txt"
    readme.write_text(
        "\n".join(
            [
                "BrainTumor-Agent NIfTI工程联调数据",
                "",
                "每个病例包含：T1、T1ce、T2、FLAIR和seg，共5个.nii.gz文件。",
                "在Web页面上传时，只选择T1/T1ce/T2/FLAIR四个文件；seg用于测试Mask叠加。",
                f"统一尺寸：{TARGET_SHAPE[0]} x {TARGET_SHAPE[1]} x {TARGET_SHAPE[2]}",
                "统一spacing：1.5 x 1.5 x 1.5 mm",
                "seg标签：0=背景，1=坏死/非增强核心，2=水肿，4=增强肿瘤。",
                "",
                "重要：该数据由程序合成，仅用于工程联调，不得用于临床或模型效果评价。",
                "真实BraTS数据请通过官方Synapse/Kaggle渠道申请并遵守数据条款。",
            ]
        )
        + "\n",
        encoding="utf-8-sig",
    )
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample_data/nifti_demo"),
        help="输出目录",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in create_demo_dataset(args.output):
        print(path)


if __name__ == "__main__":
    main()
