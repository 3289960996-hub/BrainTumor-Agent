"""nnU-Net V2 BraTS 2021 GPU训练入口。"""

import argparse
import os
import subprocess
from collections.abc import Sequence

from segmentation.prepare_dataset import (
    DEFAULT_CONFIGURATION,
    DEFAULT_DATASET_ID,
    DEFAULT_PLANS,
    SegmentationSetupError,
    configure_nnunet_environment,
    default_nnunet_root,
    run_command,
)


def normalize_folds(folds: Sequence[str]) -> tuple[str, ...]:
    """检查五折交叉验证或all折参数。"""

    normalized: list[str] = []
    for fold in folds:
        value = str(fold).lower()
        if value == "all":
            normalized.append(value)
            continue
        try:
            numeric = int(value)
        except ValueError as exc:
            raise SegmentationSetupError(f"fold必须为0至4或all：{fold}") from exc
        if numeric < 0 or numeric > 4:
            raise SegmentationSetupError(f"fold必须为0至4：{numeric}")
        normalized.append(str(numeric))

    if "all" in normalized and len(normalized) > 1:
        raise SegmentationSetupError("all折不能与0至4折同时指定")
    return tuple(dict.fromkeys(normalized))


def build_train_command(
    dataset_id: int,
    configuration: str,
    fold: str,
    trainer: str = "nnUNetTrainer",
    plans: str = DEFAULT_PLANS,
    device: str = "cuda",
    num_gpus: int = 1,
    save_npz: bool = False,
    continue_training: bool = False,
    validation_only: bool = False,
    validation_with_best: bool = False,
    disable_checkpointing: bool = False,
    pretrained_weights: str | None = None,
) -> list[str]:
    """构建单个fold的nnUNetv2_train命令。"""

    command = [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        fold,
        "-tr",
        trainer,
        "-p",
        plans,
        "-device",
        device,
        "-num_gpus",
        str(num_gpus),
    ]
    if save_npz:
        command.append("--npz")
    if continue_training:
        command.append("--c")
    if validation_only:
        command.append("--val")
    if validation_with_best:
        command.append("--val_best")
    if disable_checkpointing:
        command.append("--disable_checkpointing")
    if pretrained_weights is not None:
        command.extend(["-pretrained_weights", pretrained_weights])
    return command


def configure_device(
    device: str,
    gpu_ids: str,
    num_gpus: int,
    dry_run: bool,
) -> None:
    """设置可见GPU并在实际训练前检查CUDA。"""

    if device not in {"cuda", "cpu", "mps"}:
        raise SegmentationSetupError("device必须为cuda、cpu或mps")
    if num_gpus < 1:
        raise SegmentationSetupError("num_gpus必须大于或等于1")
    if device != "cuda":
        if num_gpus != 1:
            raise SegmentationSetupError("多GPU训练仅支持cuda")
        return

    visible_ids = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    if not visible_ids:
        raise SegmentationSetupError("使用cuda时必须至少指定一个gpu-id")
    if num_gpus > len(visible_ids):
        raise SegmentationSetupError(
            f"num_gpus={num_gpus}超过CUDA_VISIBLE_DEVICES数量{len(visible_ids)}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_ids)

    if dry_run:
        return
    import torch

    if not torch.cuda.is_available():
        raise SegmentationSetupError("PyTorch未检测到可用CUDA GPU")
    if torch.cuda.device_count() < num_gpus:
        raise SegmentationSetupError(
            f"当前进程仅检测到{torch.cuda.device_count()}张GPU，"
            f"无法启动{num_gpus}卡训练"
        )
    print(f"CUDA设备：{[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")


def train_folds(
    nnunet_root: str,
    dataset_id: int,
    configuration: str,
    folds: Sequence[str],
    trainer: str,
    plans: str,
    device: str,
    gpu_ids: str,
    num_gpus: int,
    save_npz: bool,
    continue_training: bool,
    validation_only: bool,
    validation_with_best: bool,
    disable_checkpointing: bool,
    pretrained_weights: str | None,
    dry_run: bool,
) -> None:
    """按顺序训练指定fold，默认使用GPU。"""

    paths = configure_nnunet_environment(nnunet_root)
    normalized_folds = normalize_folds(folds)
    configure_device(device, gpu_ids, num_gpus, dry_run)

    if not dry_run:
        preprocessed_matches = list(
            paths.preprocessed.glob(f"Dataset{dataset_id:03d}_*")
        )
        if len(preprocessed_matches) != 1:
            raise SegmentationSetupError(
                f"nnUNet_preprocessed中应存在唯一Dataset{dataset_id:03d}_*；"
                "请先运行prepare_dataset.py --plan-and-preprocess"
            )

    for fold in normalized_folds:
        command = build_train_command(
            dataset_id=dataset_id,
            configuration=configuration,
            fold=fold,
            trainer=trainer,
            plans=plans,
            device=device,
            num_gpus=num_gpus,
            save_npz=save_npz,
            continue_training=continue_training,
            validation_only=validation_only,
            validation_with_best=validation_with_best,
            disable_checkpointing=disable_checkpointing,
            pretrained_weights=pretrained_weights,
        )
        print(f"开始处理fold={fold}")
        run_command(command, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    """创建GPU训练命令行参数。"""

    parser = argparse.ArgumentParser(description="训练BraTS 2021 nnU-Net V2模型。")
    parser.add_argument(
        "--nnunet-root",
        default=str(default_nnunet_root()),
        help="nnU-Net工作根目录",
    )
    parser.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="训练fold，例如--folds 0 1 2 3 4或--folds all",
    )
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default=DEFAULT_PLANS)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="CUDA_VISIBLE_DEVICES值，例如0或0,1",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="单个fold使用的GPU数")
    parser.add_argument("--npz", action="store_true", help="保存验证概率用于配置集成")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validation-with-best", action="store_true")
    parser.add_argument("--disable-checkpointing", action="store_true")
    parser.add_argument("--pretrained-weights", default=None)
    parser.add_argument("--dry-run", action="store_true", help="仅打印训练命令")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行nnU-Net训练或验证。"""

    args = build_parser().parse_args(argv)
    try:
        if args.continue_training and args.validation_only:
            raise SegmentationSetupError(
                "--continue-training不能与--validation-only同时使用"
            )
        if args.pretrained_weights and args.continue_training:
            raise SegmentationSetupError(
                "--pretrained-weights不能与--continue-training同时使用"
            )
        train_folds(
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            configuration=args.configuration,
            folds=args.folds,
            trainer=args.trainer,
            plans=args.plans,
            device=args.device,
            gpu_ids=args.gpu_ids,
            num_gpus=args.num_gpus,
            save_npz=args.npz,
            continue_training=args.continue_training,
            validation_only=args.validation_only,
            validation_with_best=args.validation_with_best,
            disable_checkpointing=args.disable_checkpointing,
            pretrained_weights=args.pretrained_weights,
            dry_run=args.dry_run,
        )
    except (SegmentationSetupError, OSError, subprocess.CalledProcessError) as exc:
        print(f"训练失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
