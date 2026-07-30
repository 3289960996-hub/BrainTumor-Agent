"""BraTS 2021 nnU-Net V2 GPU推理与标签还原。"""

import argparse
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from data_process.constants import NNUNET_CHANNELS, REQUIRED_MODALITIES
from data_process.discovery import discover_study
from data_process.exceptions import DataProcessError
from segmentation.prepare_dataset import (
    DEFAULT_CONFIGURATION,
    DEFAULT_DATASET_ID,
    DEFAULT_PLANS,
    OUTPUT_LABEL_PROFILES,
    OutputLabelProfile,
    SegmentationSetupError,
    configure_nnunet_environment,
    default_nnunet_root,
    find_brats_case_dirs,
    restore_segmentation_file,
    run_command,
)
from segmentation.train import configure_device, normalize_folds


def _strip_nifti_suffix(filename: str) -> str:
    """移除.nii或.nii.gz扩展名。"""

    lower_name = filename.lower()
    if lower_name.endswith(".nii.gz"):
        return filename[:-7]
    if lower_name.endswith(".nii"):
        return filename[:-4]
    raise SegmentationSetupError(f"不是NIfTI文件：{filename}")


def _discover_nnunet_case_ids(input_dir: Path) -> list[str]:
    """检查已采用_0000至_0003命名的推理目录。"""

    channel_zero_files = sorted(input_dir.glob("*_0000.nii.gz"))
    if not channel_zero_files:
        return []

    case_ids: list[str] = []
    for channel_zero in channel_zero_files:
        stem = _strip_nifti_suffix(channel_zero.name)
        case_id = stem[:-5]
        for modality in REQUIRED_MODALITIES:
            expected = input_dir / f"{case_id}_{NNUNET_CHANNELS[modality]}.nii.gz"
            if not expected.is_file():
                raise SegmentationSetupError(
                    f"nnU-Net推理输入缺少文件：{expected.name}"
                )
        case_ids.append(case_id)
    return case_ids


def prepare_inference_inputs(
    input_dir: str | Path,
    staging_dir: str | Path,
    overwrite: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    """接受nnU-Net格式目录或原始BraTS病例目录，并返回标准推理目录。"""

    source = Path(input_dir).expanduser().resolve()
    if not source.is_dir():
        raise SegmentationSetupError(f"推理输入目录不存在：{source}")

    nnunet_case_ids = _discover_nnunet_case_ids(source)
    if nnunet_case_ids:
        return source, tuple(nnunet_case_ids)

    target = Path(staging_dir).expanduser().resolve()
    case_dirs = find_brats_case_dirs(source)
    discovered_cases: list[tuple[str, dict]] = []
    for case_dir in case_dirs:
        try:
            manifest = discover_study(case_dir)
        except DataProcessError as exc:
            raise SegmentationSetupError(f"推理病例不合法：{case_dir}：{exc}") from exc
        discovered_cases.append((manifest.case_id, manifest.by_modality()))

    target_files = [
        target / f"{case_id}_{NNUNET_CHANNELS[modality]}.nii.gz"
        for case_id, _ in discovered_cases
        for modality in REQUIRED_MODALITIES
    ]
    existing = [path for path in target_files if path.exists()]
    if existing and not overwrite:
        raise SegmentationSetupError(
            f"推理暂存文件已存在：{existing[0]}；如需更新请使用--overwrite"
        )
    target.mkdir(parents=True, exist_ok=True)

    for case_id, modality_map in discovered_cases:
        for modality in REQUIRED_MODALITIES:
            shutil.copy2(
                modality_map[modality].path,
                target / f"{case_id}_{NNUNET_CHANNELS[modality]}.nii.gz",
            )
    return target, tuple(case_id for case_id, _ in discovered_cases)


def build_predict_command(
    input_dir: Path,
    output_dir: Path,
    dataset_id: int,
    configuration: str,
    plans: str,
    trainer: str,
    folds: Sequence[str],
    device: str,
    checkpoint: str,
    step_size: float,
    preprocessing_processes: int,
    export_processes: int,
    save_probabilities: bool = False,
    disable_tta: bool = False,
    continue_prediction: bool = False,
    not_on_device: bool = False,
) -> list[str]:
    """构建nnUNetv2_predict命令。"""

    if step_size <= 0.0 or step_size > 1.0:
        raise SegmentationSetupError("step_size必须位于(0, 1]范围")
    if preprocessing_processes < 1 or export_processes < 1:
        raise SegmentationSetupError("预处理和导出进程数必须大于或等于1")

    command = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(dataset_id),
        "-c",
        configuration,
        "-p",
        plans,
        "-tr",
        trainer,
        "-f",
        *folds,
        "-device",
        device,
        "-chk",
        checkpoint,
        "-step_size",
        str(step_size),
        "-npp",
        str(preprocessing_processes),
        "-nps",
        str(export_processes),
    ]
    if save_probabilities:
        command.append("--save_probabilities")
    if disable_tta:
        command.append("--disable_tta")
    if continue_prediction:
        command.append("--continue_prediction")
    if not_on_device:
        command.append("--not_on_device")
    return command


def restore_prediction_folder(
    nnunet_prediction_dir: str | Path,
    brats_prediction_dir: str | Path,
    overwrite: bool = False,
    output_label_profile: OutputLabelProfile = "standard_nnunet",
) -> tuple[Path, ...]:
    """批量将nnU-Net标签0/1/2/3还原为BraTS标签0/2/1/4。"""

    source_dir = Path(nnunet_prediction_dir).expanduser().resolve()
    target_dir = Path(brats_prediction_dir).expanduser().resolve()
    predictions = sorted(source_dir.glob("*.nii.gz"))
    if not predictions:
        raise SegmentationSetupError(f"未找到nnU-Net预测NIfTI：{source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    restored: list[Path] = []
    for source in predictions:
        target = target_dir / source.name
        if target.exists() and not overwrite:
            raise SegmentationSetupError(
                f"BraTS标签预测已存在：{target}；如需覆盖请使用--overwrite"
            )
        restored.append(
            restore_segmentation_file(
                source,
                target,
                output_label_profile=output_label_profile,
            )
        )
    return tuple(restored)


def build_parser() -> argparse.ArgumentParser:
    """创建GPU推理命令行参数。"""

    parser = argparse.ArgumentParser(
        description="使用nnU-Net V2对BraTS四模态MRI执行GPU推理。",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="原始BraTS病例根目录或已采用_0000至_0003命名的目录",
    )
    parser.add_argument("--output-dir", required=True, help="推理输出根目录")
    parser.add_argument(
        "--nnunet-root",
        default=str(default_nnunet_root()),
        help="nnU-Net工作根目录",
    )
    parser.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument("--plans", default=DEFAULT_PLANS)
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--preprocessing-processes", type=int, default=3)
    parser.add_argument("--export-processes", type=int, default=3)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--continue-prediction", action="store_true")
    parser.add_argument(
        "--not-on-device",
        action="store_true",
        help="显存不足时将更多步骤放到CPU",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-restore-brats-labels",
        action="store_true",
        help="仅保留nnU-Net内部0/1/2/3标签",
    )
    parser.add_argument(
        "--output-label-profile",
        choices=OUTPUT_LABEL_PROFILES,
        default="standard_nnunet",
        help=(
            "模型输出标签配置；Dataset002_BRATS19权重使用brats19_preserved"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印推理命令")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行nnU-Net推理并默认还原BraTS标签。"""

    args = build_parser().parse_args(argv)
    try:
        configure_nnunet_environment(args.nnunet_root)
        folds = normalize_folds(args.folds)
        configure_device(
            device=args.device,
            gpu_ids=args.gpu_id,
            num_gpus=1,
            dry_run=args.dry_run,
        )

        output_root = Path(args.output_dir).expanduser().resolve()
        staging_dir = output_root / "nnunet_input"
        nnunet_output = output_root / "nnunet_predictions"
        brats_output = output_root / "brats_predictions"

        if args.dry_run:
            source_path = Path(args.input_dir).expanduser().resolve()
            input_path = (
                source_path
                if source_path.is_dir() and _discover_nnunet_case_ids(source_path)
                else staging_dir
            )
        else:
            input_path, case_ids = prepare_inference_inputs(
                input_dir=args.input_dir,
                staging_dir=staging_dir,
                overwrite=args.overwrite,
            )
            print(f"待推理病例数：{len(case_ids)}")

        command = build_predict_command(
            input_dir=input_path,
            output_dir=nnunet_output,
            dataset_id=args.dataset_id,
            configuration=args.configuration,
            plans=args.plans,
            trainer=args.trainer,
            folds=folds,
            device=args.device,
            checkpoint=args.checkpoint,
            step_size=args.step_size,
            preprocessing_processes=args.preprocessing_processes,
            export_processes=args.export_processes,
            save_probabilities=args.save_probabilities,
            disable_tta=args.disable_tta,
            continue_prediction=args.continue_prediction,
            not_on_device=args.not_on_device,
        )
        run_command(command, dry_run=args.dry_run)

        if not args.dry_run and not args.no_restore_brats_labels:
            restored = restore_prediction_folder(
                nnunet_prediction_dir=nnunet_output,
                brats_prediction_dir=brats_output,
                overwrite=args.overwrite,
                output_label_profile=args.output_label_profile,
            )
            print(f"BraTS标签预测目录：{brats_output}，病例数={len(restored)}")
    except (SegmentationSetupError, OSError, subprocess.CalledProcessError) as exc:
        print(f"推理失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
