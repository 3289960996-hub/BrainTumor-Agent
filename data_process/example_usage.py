r"""Python API运行示例。

示例：
python -m data_process.example_usage ^
  --input-dir D:\BraTS2021\BraTS2021_00000 ^
  --output-dir D:\BraTS2021_processed
"""

import argparse
from pathlib import Path

from data_process.processor import BraTSDataProcessor


def process_example(input_dir: Path, output_dir: Path) -> None:
    """演示使用Python API处理一个BraTS病例。"""

    processor = BraTSDataProcessor()
    result = processor.process_case(
        case_dir=input_dir,
        output_root=output_dir,
    )
    print(f"病例{result.case_id}处理完成，输出目录：{result.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainTumor-Agent数据处理API示例")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    process_example(arguments.input_dir, arguments.output_dir)
