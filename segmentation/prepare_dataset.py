"""将BraTS 2021训练集转换为nnU-Net V2格式。

主要职责：
1. 配置nnUNet_raw、nnUNet_preprocessed和nnUNet_results；
2. 复制T1、T1ce、T2、FLAIR到固定通道0000至0003；
3. 将BraTS标签0/1/2/4转换为nnU-Net连续标签0/2/1/3；
4. 自动生成支持WT、TC、ET区域训练的dataset.json；
5. 可选执行GPU训练前的规划与预处理。
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from data_process.constants import NNUNET_CHANNELS, REQUIRED_MODALITIES, MRIModality
from data_process.discovery import discover_study
from data_process.exceptions import DataProcessError

DEFAULT_DATASET_ID = 137
DEFAULT_DATASET_NAME = "BraTS2021"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_PLANNER = "nnUNetPlannerResEncM"
DEFAULT_PLANS = "nnUNetResEncUNetMPlans"

BRATS_LABELS = frozenset({0, 1, 2, 4})
NNUNET_LABELS = frozenset({0, 1, 2, 3})
BRATS19_PRESERVED_LABELS = frozenset({0, 1, 2, 3, 4})

OutputLabelProfile = Literal["standard_nnunet", "brats19_preserved"]
OUTPUT_LABEL_PROFILES: tuple[OutputLabelProfile, ...] = (
    "standard_nnunet",
    "brats19_preserved",
)


class SegmentationSetupError(RuntimeError):
    """nnU-Net环境或BraTS转换不满足要求。"""


@dataclass(frozen=True, slots=True)
class NnUNetPaths:
    """nnU-Net V2要求的三个工作目录。"""

    root: Path
    raw: Path
    preprocessed: Path
    results: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "NnUNetPaths":
        """从统一根目录构造nnU-Net路径。"""

        resolved_root = Path(root).expanduser().resolve()
        return cls(
            root=resolved_root,
            raw=resolved_root / "nnUNet_raw",
            preprocessed=resolved_root / "nnUNet_preprocessed",
            results=resolved_root / "nnUNet_results",
        )

    def configure_environment(self, create: bool = True) -> None:
        """设置当前进程及其子进程使用的nnU-Net环境变量。"""

        if create:
            self.raw.mkdir(parents=True, exist_ok=True)
            self.preprocessed.mkdir(parents=True, exist_ok=True)
            self.results.mkdir(parents=True, exist_ok=True)
        os.environ["nnUNet_raw"] = str(self.raw)
        os.environ["nnUNet_preprocessed"] = str(self.preprocessed)
        os.environ["nnUNet_results"] = str(self.results)


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """完成转换的nnU-Net数据集信息。"""

    dataset_id: int
    dataset_name: str
    dataset_dir: Path
    num_training_cases: int
    dataset_json_path: Path
    nnunet_paths: NnUNetPaths


def default_nnunet_root() -> Path:
    """返回命令行默认nnU-Net根目录。"""

    return Path(os.environ.get("NNUNET_ROOT", "./runtime/nnunet"))


def configure_nnunet_environment(root: str | Path) -> NnUNetPaths:
    """创建并导出nnU-Net V2工作目录。"""

    paths = NnUNetPaths.from_root(root)
    paths.configure_environment(create=True)
    return paths


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    """生成DatasetXXX_Name目录名并检查参数安全性。"""

    if dataset_id < 1 or dataset_id > 999:
        raise SegmentationSetupError("dataset_id必须位于1至999")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", dataset_name):
        raise SegmentationSetupError(
            "dataset_name必须以字母开头，且只能包含字母、数字和下划线"
        )
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def convert_brats_to_nnunet_labels(
    segmentation: NDArray[np.integer | np.floating],
) -> NDArray[np.uint8]:
    """将BraTS标签0/1/2/4转换为nnU-Net连续标签0/2/1/3。

    映射与nnU-Net官方Dataset137_BraTS21转换器保持一致：
    BraTS ED(2)->1，NCR/NET(1)->2，ET(4)->3。
    """

    labels = np.rint(np.asarray(segmentation)).astype(np.int16)
    unique_labels = set(int(value) for value in np.unique(labels))
    unexpected = unique_labels - BRATS_LABELS
    if unexpected:
        raise SegmentationSetupError(f"BraTS mask包含未知标签：{sorted(unexpected)}")

    converted = np.zeros(labels.shape, dtype=np.uint8)
    converted[labels == 2] = 1
    converted[labels == 1] = 2
    converted[labels == 4] = 3
    return converted


def convert_nnunet_to_brats_labels(
    segmentation: NDArray[np.integer | np.floating],
) -> NDArray[np.uint8]:
    """将nnU-Net连续标签0/1/2/3还原为BraTS标签0/2/1/4。"""

    labels = np.rint(np.asarray(segmentation)).astype(np.int16)
    unique_labels = set(int(value) for value in np.unique(labels))
    unexpected = unique_labels - NNUNET_LABELS
    if unexpected:
        raise SegmentationSetupError(f"nnU-Net预测包含未知标签：{sorted(unexpected)}")

    converted = np.zeros(labels.shape, dtype=np.uint8)
    converted[labels == 1] = 2
    converted[labels == 2] = 1
    converted[labels == 3] = 4
    return converted


def convert_brats19_preserved_to_brats_labels(
    segmentation: NDArray[np.integer | np.floating],
) -> NDArray[np.uint8]:
    """还原Dataset002_BRATS19自定义五类输出。

    该模型的dataset.json定义为：
    0=background，1=edema，2=nonenhancing，3=empty，4=enhancing。
    输出转为标准BraTS标签0/1/2/4，其中未使用的empty类3按背景处理。
    """

    labels = np.rint(np.asarray(segmentation)).astype(np.int16)
    unique_labels = set(int(value) for value in np.unique(labels))
    unexpected = unique_labels - BRATS19_PRESERVED_LABELS
    if unexpected:
        raise SegmentationSetupError(
            f"Dataset002_BRATS19预测包含未知标签：{sorted(unexpected)}"
        )

    converted = np.zeros(labels.shape, dtype=np.uint8)
    converted[labels == 1] = 2
    converted[labels == 2] = 1
    converted[labels == 4] = 4
    return converted


def convert_model_output_to_brats_labels(
    segmentation: NDArray[np.integer | np.floating],
    output_label_profile: OutputLabelProfile = "standard_nnunet",
) -> NDArray[np.uint8]:
    """按模型标签配置将预测统一成标准BraTS标签0/1/2/4。"""

    if output_label_profile == "standard_nnunet":
        return convert_nnunet_to_brats_labels(segmentation)
    if output_label_profile == "brats19_preserved":
        return convert_brats19_preserved_to_brats_labels(segmentation)
    raise SegmentationSetupError(
        f"不支持的模型输出标签配置：{output_label_profile}"
    )


def convert_segmentation_file(source: Path, target: Path) -> Path:
    """转换标签并保留原始NIfTI物理空间信息。"""

    image = sitk.ReadImage(str(source))
    array = sitk.GetArrayFromImage(image)
    converted = convert_brats_to_nnunet_labels(array)
    output = sitk.GetImageFromArray(converted)
    output.CopyInformation(image)
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(target), useCompression=True)
    return target


def restore_segmentation_file(
    source: Path,
    target: Path,
    output_label_profile: OutputLabelProfile = "standard_nnunet",
) -> Path:
    """按模型配置还原BraTS标签并保留空间信息。"""

    image = sitk.ReadImage(str(source))
    array = sitk.GetArrayFromImage(image)
    converted = convert_model_output_to_brats_labels(
        array,
        output_label_profile=output_label_profile,
    )
    output = sitk.GetImageFromArray(converted)
    output.CopyInformation(image)
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(target), useCompression=True)
    return target


def _read_geometry(
    path: Path,
) -> tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """只读取NIfTI头信息，避免为几何检查加载四套完整体素。"""

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return (
        tuple(int(value) for value in reader.GetSize()),
        tuple(float(value) for value in reader.GetSpacing()),
        tuple(float(value) for value in reader.GetOrigin()),
        tuple(float(value) for value in reader.GetDirection()),
    )


def _validate_geometry(reference_path: Path, candidate_path: Path) -> None:
    """检查影像或标签与T1参考空间完全对应。"""

    reference = _read_geometry(reference_path)
    candidate = _read_geometry(candidate_path)
    names = ("size", "spacing", "origin", "direction")
    tolerances = (0.0, 1e-5, 1e-4, 1e-4)
    errors: list[str] = []
    for name, reference_value, candidate_value, tolerance in zip(
        names,
        reference,
        candidate,
        tolerances,
        strict=True,
    ):
        if tolerance == 0.0:
            matches = candidate_value == reference_value
        else:
            matches = bool(
                np.allclose(candidate_value, reference_value, rtol=0.0, atol=tolerance)
            )
        if not matches:
            errors.append(
                f"{candidate_path.name}的{name}={candidate_value}，"
                f"T1参考值={reference_value}"
            )
    if errors:
        raise SegmentationSetupError("；".join(errors))


def _find_segmentation(case_dir: Path, case_id: str) -> Path:
    """查找单个BraTS训练标签。"""

    candidates = [
        path
        for path in case_dir.iterdir()
        if path.is_file()
        and path.name.lower()
        in {f"{case_id.lower()}_seg.nii.gz", f"{case_id.lower()}_seg.nii"}
    ]
    if len(candidates) != 1:
        raise SegmentationSetupError(
            f"病例{case_id}应包含且仅包含一个*_seg.nii.gz，实际找到{len(candidates)}个"
        )
    return candidates[0]


def find_brats_case_dirs(brats_root: str | Path) -> list[Path]:
    """发现BraTS训练根目录下的病例文件夹。"""

    root = Path(brats_root).expanduser().resolve()
    if not root.is_dir():
        raise SegmentationSetupError(f"BraTS训练目录不存在：{root}")

    def contains_t1(directory: Path) -> bool:
        return any(
            path.is_file()
            and (
                path.name.lower().endswith("_t1.nii.gz")
                or path.name.lower().endswith("_t1.nii")
            )
            for path in directory.iterdir()
        )

    if contains_t1(root):
        return [root]
    case_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and contains_t1(path)),
        key=lambda path: path.name,
    )
    if not case_dirs:
        raise SegmentationSetupError(f"未发现BraTS病例目录：{root}")
    return case_dirs


def _prepare_one_case(
    case_dir: Path,
    images_tr: Path,
    labels_tr: Path,
) -> str:
    """转换一个病例，返回标准病例ID。"""

    try:
        manifest = discover_study(case_dir)
    except DataProcessError as exc:
        raise SegmentationSetupError(f"病例目录不合法：{case_dir}：{exc}") from exc

    case_id = manifest.case_id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id):
        raise SegmentationSetupError(f"病例ID包含不安全字符：{case_id}")

    modality_map = manifest.by_modality()
    reference_path = modality_map[MRIModality.T1].path
    segmentation_path = _find_segmentation(case_dir, case_id)

    for modality in REQUIRED_MODALITIES[1:]:
        _validate_geometry(reference_path, modality_map[modality].path)
    _validate_geometry(reference_path, segmentation_path)

    for modality in REQUIRED_MODALITIES:
        target = images_tr / f"{case_id}_{NNUNET_CHANNELS[modality]}.nii.gz"
        shutil.copy2(modality_map[modality].path, target)
    convert_segmentation_file(
        source=segmentation_path,
        target=labels_tr / f"{case_id}.nii.gz",
    )
    return case_id


def generate_dataset_json(
    dataset_dir: Path,
    num_training_cases: int,
    dataset_name: str,
) -> Path:
    """生成nnU-Net V2区域训练dataset.json。"""

    payload = {
        "name": dataset_name,
        "description": "BraTS 2021 four-modal MRI brain tumor segmentation",
        "reference": "https://www.med.upenn.edu/cbica/brats2021/",
        "licence": "See the BraTS 2021 data usage agreement",
        "channel_names": {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR",
        },
        "labels": {
            "background": 0,
            "whole tumor": [1, 2, 3],
            "tumor core": [2, 3],
            "enhancing tumor": [3],
        },
        "regions_class_order": [1, 2, 3],
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    target = dataset_dir / "dataset.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def prepare_brats_dataset(
    brats_root: str | Path,
    nnunet_root: str | Path,
    dataset_id: int = DEFAULT_DATASET_ID,
    dataset_name: str = DEFAULT_DATASET_NAME,
    workers: int = 4,
    overwrite: bool = False,
) -> PreparedDataset:
    """转换完整BraTS训练集并生成dataset.json。"""

    if workers < 1:
        raise SegmentationSetupError("workers必须大于或等于1")

    paths = configure_nnunet_environment(nnunet_root)
    folder_name = dataset_folder_name(dataset_id, dataset_name)
    dataset_dir = paths.raw / folder_name

    id_collisions = [
        path
        for path in paths.raw.glob(f"Dataset{dataset_id:03d}_*")
        if path.resolve() != dataset_dir.resolve()
    ]
    if id_collisions:
        raise SegmentationSetupError(
            f"dataset_id {dataset_id}已被占用：{id_collisions[0].name}"
        )

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        if not overwrite:
            raise SegmentationSetupError(
                f"目标数据集已存在：{dataset_dir}；如需重建请显式使用--overwrite"
            )
        if dataset_dir.parent.resolve() != paths.raw.resolve():
            raise SegmentationSetupError("拒绝删除nnUNet_raw之外的目录")
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_dirs = find_brats_case_dirs(brats_root)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        case_ids = list(
            executor.map(
                lambda directory: _prepare_one_case(directory, images_tr, labels_tr),
                case_dirs,
            )
        )
    if len(case_ids) != len(set(case_ids)):
        raise SegmentationSetupError("发现重复病例ID")

    dataset_json_path = generate_dataset_json(
        dataset_dir=dataset_dir,
        num_training_cases=len(case_ids),
        dataset_name=dataset_name,
    )
    return PreparedDataset(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        num_training_cases=len(case_ids),
        dataset_json_path=dataset_json_path,
        nnunet_paths=paths,
    )


def format_command(command: Sequence[str]) -> str:
    """按当前操作系统格式化命令，仅用于日志显示。"""

    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def run_command(command: Sequence[str], dry_run: bool = False) -> None:
    """安全执行nnU-Net命令，不使用shell字符串拼接。"""

    print(f"$ {format_command(command)}")
    if dry_run:
        return
    if shutil.which(command[0]) is None:
        raise SegmentationSetupError(
            f"找不到命令{command[0]}，请先安装nnunetv2并激活正确的Python环境"
        )
    subprocess.run(list(command), check=True)


def build_plan_and_preprocess_command(
    dataset_id: int,
    configuration: str = DEFAULT_CONFIGURATION,
    planner: str = DEFAULT_PLANNER,
    processes: int | None = None,
    no_progress_bar: bool = False,
) -> list[str]:
    """构建nnU-Net规划与预处理命令。"""

    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(dataset_id),
        "--verify_dataset_integrity",
        "-c",
        configuration,
        "-pl",
        planner,
    ]
    if processes is not None:
        command.extend(["-np", str(processes)])
    if no_progress_bar:
        command.append("--no_pbar")
    return command


def build_parser() -> argparse.ArgumentParser:
    """创建数据转换命令行参数。"""

    parser = argparse.ArgumentParser(
        description="将BraTS 2021转换为nnU-Net V2四模态区域训练格式。",
    )
    parser.add_argument("--brats-dir", required=True, help="BraTS 2021训练集根目录")
    parser.add_argument(
        "--nnunet-root",
        default=str(default_nnunet_root()),
        help="nnU-Net工作根目录",
    )
    parser.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--workers", type=int, default=4, help="并行转换线程数")
    parser.add_argument("--overwrite", action="store_true", help="显式重建目标数据集")
    parser.add_argument(
        "--plan-and-preprocess",
        action="store_true",
        help="转换完成后执行规划、完整性检查和预处理",
    )
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument("--planner", default=DEFAULT_PLANNER)
    parser.add_argument("--preprocess-processes", type=int, default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印规划与预处理命令；数据转换仍会执行",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行BraTS数据转换。"""

    args = build_parser().parse_args(argv)
    try:
        prepared = prepare_brats_dataset(
            brats_root=args.brats_dir,
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        print(
            f"转换完成：{prepared.dataset_dir}，"
            f"训练病例数={prepared.num_training_cases}"
        )
        print(f"dataset.json：{prepared.dataset_json_path}")
        if args.plan_and_preprocess:
            command = build_plan_and_preprocess_command(
                dataset_id=args.dataset_id,
                configuration=args.configuration,
                planner=args.planner,
                processes=args.preprocess_processes,
                no_progress_bar=args.no_progress_bar,
            )
            run_command(command, dry_run=args.dry_run)
    except (SegmentationSetupError, OSError, subprocess.CalledProcessError) as exc:
        print(f"数据准备失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
