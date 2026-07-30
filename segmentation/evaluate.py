"""计算BraTS Whole Tumor、Tumor Core和Enhancing Tumor Dice。"""

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from data_process.io import read_nifti
from segmentation.prepare_dataset import (
    BRATS_LABELS,
    NNUNET_LABELS,
    SegmentationSetupError,
)


@dataclass(frozen=True, slots=True)
class CaseDice:
    """单病例三个BraTS区域的Dice。"""

    case_id: str
    whole_tumor: float
    tumor_core: float
    enhancing_tumor: float


@dataclass(frozen=True, slots=True)
class DiceSummary:
    """数据集平均Dice。"""

    num_cases: int
    whole_tumor: float
    tumor_core: float
    enhancing_tumor: float


def dice_score(
    prediction: NDArray[np.bool_],
    target: NDArray[np.bool_],
) -> float:
    """计算二值Dice；预测和真值都为空时记为1。"""

    prediction_bool = np.asarray(prediction, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    denominator = int(prediction_bool.sum()) + int(target_bool.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(prediction_bool, target_bool).sum())
    return 2.0 * intersection / denominator


def labels_to_regions(
    labels: NDArray[np.integer | np.floating],
    label_space: str,
) -> dict[str, NDArray[np.bool_]]:
    """将BraTS或nnU-Net标签转换为WT、TC和ET二值区域。"""

    values = np.rint(np.asarray(labels)).astype(np.int16)
    unique_values = set(int(value) for value in np.unique(values))

    if label_space == "brats":
        unexpected = unique_values - BRATS_LABELS
        if unexpected:
            raise SegmentationSetupError(
                f"BraTS标签空间包含未知值：{sorted(unexpected)}"
            )
        return {
            "whole_tumor": np.isin(values, (1, 2, 4)),
            "tumor_core": np.isin(values, (1, 4)),
            "enhancing_tumor": values == 4,
        }
    if label_space == "nnunet":
        unexpected = unique_values - NNUNET_LABELS
        if unexpected:
            raise SegmentationSetupError(
                f"nnU-Net标签空间包含未知值：{sorted(unexpected)}"
            )
        return {
            "whole_tumor": np.isin(values, (1, 2, 3)),
            "tumor_core": np.isin(values, (2, 3)),
            "enhancing_tumor": values == 3,
        }
    raise SegmentationSetupError("label_space必须为brats或nnunet")


def _strip_nifti_suffix(filename: str) -> str:
    """移除NIfTI扩展名。"""

    lower_name = filename.lower()
    if lower_name.endswith(".nii.gz"):
        return filename[:-7]
    if lower_name.endswith(".nii"):
        return filename[:-4]
    raise SegmentationSetupError(f"不是NIfTI文件：{filename}")


def _label_case_id(path: Path) -> str | None:
    """从标签文件名提取病例ID，并跳过四模态输入文件。"""

    stem = _strip_nifti_suffix(path.name)
    if any(stem.endswith(f"_{channel}") for channel in ("0000", "0001", "0002", "0003")):
        return None
    if any(stem.lower().endswith(f"_{modality}") for modality in ("t1", "t1ce", "t2", "flair")):
        return None
    if stem.lower().endswith("_seg"):
        return stem[:-4]
    return stem


def discover_label_files(directory: str | Path) -> dict[str, Path]:
    """发现扁平目录或BraTS病例子目录中的标签文件。"""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise SegmentationSetupError(f"标签目录不存在：{root}")

    candidates = sorted(root.glob("*.nii")) + sorted(root.glob("*.nii.gz"))
    if not candidates:
        candidates = sorted(root.glob("*/*.nii")) + sorted(root.glob("*/*.nii.gz"))

    labels: dict[str, Path] = {}
    for path in candidates:
        case_id = _label_case_id(path)
        if case_id is None:
            continue
        if case_id in labels:
            raise SegmentationSetupError(
                f"病例{case_id}存在重复标签：{labels[case_id]}和{path}"
            )
        labels[case_id] = path
    if not labels:
        raise SegmentationSetupError(f"未发现标签NIfTI：{root}")
    return labels


def _validate_prediction_geometry(
    ground_truth_path: Path,
    prediction_path: Path,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """读取预测和真值，并验证尺寸、spacing及affine。"""

    ground_truth = read_nifti(ground_truth_path)
    prediction = read_nifti(prediction_path)
    errors: list[str] = []
    if prediction.geometry.shape != ground_truth.geometry.shape:
        errors.append(
            f"shape预测={prediction.geometry.shape}，真值={ground_truth.geometry.shape}"
        )
    if not np.allclose(
        prediction.geometry.spacing,
        ground_truth.geometry.spacing,
        rtol=0.0,
        atol=1e-5,
    ):
        errors.append(
            f"spacing预测={prediction.geometry.spacing}，"
            f"真值={ground_truth.geometry.spacing}"
        )
    if not np.allclose(
        prediction.geometry.affine,
        ground_truth.geometry.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        errors.append("affine不一致")
    if errors:
        raise SegmentationSetupError(
            f"预测{prediction_path.name}与真值空间不一致：" + "；".join(errors)
        )
    return ground_truth.data, prediction.data


def evaluate_case(
    case_id: str,
    ground_truth_path: Path,
    prediction_path: Path,
    ground_truth_label_space: str = "brats",
    prediction_label_space: str = "brats",
) -> CaseDice:
    """计算一个病例的WT、TC和ET Dice。"""

    ground_truth, prediction = _validate_prediction_geometry(
        ground_truth_path,
        prediction_path,
    )
    ground_truth_regions = labels_to_regions(ground_truth, ground_truth_label_space)
    prediction_regions = labels_to_regions(prediction, prediction_label_space)
    return CaseDice(
        case_id=case_id,
        whole_tumor=dice_score(
            prediction_regions["whole_tumor"],
            ground_truth_regions["whole_tumor"],
        ),
        tumor_core=dice_score(
            prediction_regions["tumor_core"],
            ground_truth_regions["tumor_core"],
        ),
        enhancing_tumor=dice_score(
            prediction_regions["enhancing_tumor"],
            ground_truth_regions["enhancing_tumor"],
        ),
    )


def summarize_metrics(results: Sequence[CaseDice]) -> DiceSummary:
    """计算所有病例的宏平均Dice。"""

    if not results:
        raise SegmentationSetupError("没有可汇总的评估病例")
    return DiceSummary(
        num_cases=len(results),
        whole_tumor=float(np.mean([item.whole_tumor for item in results])),
        tumor_core=float(np.mean([item.tumor_core for item in results])),
        enhancing_tumor=float(
            np.mean([item.enhancing_tumor for item in results])
        ),
    )


def evaluate_folders(
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    ground_truth_label_space: str = "brats",
    prediction_label_space: str = "brats",
) -> tuple[DiceSummary, tuple[CaseDice, ...]]:
    """按病例名匹配两个目录并计算Dice。"""

    ground_truth_files = discover_label_files(ground_truth_dir)
    prediction_files = discover_label_files(prediction_dir)
    missing_predictions = sorted(set(ground_truth_files) - set(prediction_files))
    unexpected_predictions = sorted(set(prediction_files) - set(ground_truth_files))
    if missing_predictions:
        raise SegmentationSetupError(
            f"缺少{len(missing_predictions)}个预测，例如：{missing_predictions[0]}"
        )
    if unexpected_predictions:
        raise SegmentationSetupError(
            f"存在{len(unexpected_predictions)}个无真值预测，例如："
            f"{unexpected_predictions[0]}"
        )

    results = tuple(
        evaluate_case(
            case_id=case_id,
            ground_truth_path=ground_truth_files[case_id],
            prediction_path=prediction_files[case_id],
            ground_truth_label_space=ground_truth_label_space,
            prediction_label_space=prediction_label_space,
        )
        for case_id in sorted(ground_truth_files)
    )
    return summarize_metrics(results), results


def save_evaluation(
    summary: DiceSummary,
    cases: Sequence[CaseDice],
    output_json: str | Path,
    output_csv: str | Path,
    metadata: Mapping[str, str],
) -> None:
    """保存机器可读JSON和逐病例CSV。"""

    json_path = Path(output_json).expanduser().resolve()
    csv_path = Path(output_csv).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": asdict(summary),
        "cases": [asdict(item) for item in cases],
        "metadata": dict(metadata),
        "empty_region_policy": "prediction and target both empty => Dice 1.0",
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "whole_tumor",
                "tumor_core",
                "enhancing_tumor",
            ],
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in cases)


def build_parser() -> argparse.ArgumentParser:
    """创建Dice评估命令行参数。"""

    parser = argparse.ArgumentParser(
        description="评估BraTS预测的WT、TC和ET Dice。",
    )
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument(
        "--ground-truth-label-space",
        choices=["brats", "nnunet"],
        default="brats",
    )
    parser.add_argument(
        "--prediction-label-space",
        choices=["brats", "nnunet"],
        default="brats",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行目录级Dice评估并打印三个核心指标。"""

    args = build_parser().parse_args(argv)
    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    output_json = args.output_json or prediction_dir / "evaluation.json"
    output_csv = args.output_csv or prediction_dir / "evaluation_cases.csv"
    try:
        summary, cases = evaluate_folders(
            ground_truth_dir=args.ground_truth_dir,
            prediction_dir=prediction_dir,
            ground_truth_label_space=args.ground_truth_label_space,
            prediction_label_space=args.prediction_label_space,
        )
        save_evaluation(
            summary=summary,
            cases=cases,
            output_json=output_json,
            output_csv=output_csv,
            metadata={
                "ground_truth_label_space": args.ground_truth_label_space,
                "prediction_label_space": args.prediction_label_space,
            },
        )
    except (SegmentationSetupError, OSError) as exc:
        print(f"评估失败：{exc}")
        return 1

    print(f"Cases: {summary.num_cases}")
    print(f"Whole Tumor Dice: {summary.whole_tumor:.6f}")
    print(f"Tumor Core Dice: {summary.tumor_core:.6f}")
    print(f"Enhancing Tumor Dice: {summary.enhancing_tumor:.6f}")
    print(f"JSON: {Path(output_json).expanduser().resolve()}")
    print(f"CSV: {Path(output_csv).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
