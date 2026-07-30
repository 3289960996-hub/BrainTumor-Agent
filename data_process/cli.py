"""BraTS 2021数据处理命令行入口。"""

import argparse
from collections.abc import Sequence

from data_process.exceptions import DataProcessError


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="读取、检查、归一化并保存一个BraTS 2021四模态MRI病例。",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="包含*_t1、*_t1ce、*_t2、*_flair.nii.gz的病例目录",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="processed数据根目录",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="可选病例ID；未指定时从文件名自动推断",
    )
    parser.add_argument(
        "--spacing-tolerance",
        type=float,
        default=1e-5,
        help="四模态spacing检查的绝对误差，默认1e-5",
    )
    parser.add_argument(
        "--affine-tolerance",
        type=float,
        default=1e-4,
        help="affine/origin/direction检查的绝对误差，默认1e-4",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同名processed文件",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行单病例处理并输出结果文件路径。"""

    args = build_parser().parse_args(argv)

    # 解析完参数后再加载MONAI相关模块，使--help可以快速返回。
    from data_process.processor import BraTSDataProcessor

    processor = BraTSDataProcessor(
        spacing_tolerance=args.spacing_tolerance,
        affine_tolerance=args.affine_tolerance,
    )
    try:
        result = processor.process_case(
            case_dir=args.input_dir,
            output_root=args.output_dir,
            case_id=args.case_id,
            overwrite=args.overwrite,
        )
    except DataProcessError as exc:
        print(f"处理失败：{exc}")
        return 1

    print(f"处理完成：{result.case_id}")
    for modality, path in result.modality_files.items():
        print(f"  {modality.value}: {path}")
    print(f"  metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
