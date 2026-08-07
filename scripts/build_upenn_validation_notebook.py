"""Build the standalone UPENN-GBM 10-case Colab validation notebook."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "notebooks" / (
    "BrainTumor_Agent_UPENN_10_GPU_Validation_Colab.ipynb"
)


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def main() -> None:
    cells = [
        markdown(
            """# BrainTumor-Agent: independent UPENN-GBM 10-case GPU validation

This notebook runs the five-fold Dataset002_BRATS19 ensemble on ten baseline
UPENN-GBM cases with expert-reviewed masks. UPENN-GBM is an institutional cohort
independent of the old BraTS/MSD training sources. Source DOI:
`10.7937/TCIA.709X-DN49` (CC BY 4.0).
"""
        ),
        code(
            """from google.colab import drive

drive.mount('/content/drive')
"""
        ),
        code(
            """from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/BrainTumor-Agent-Colab')
INPUT_DIR = DRIVE_ROOT / 'inputs'
DRIVE_RESULTS = DRIVE_ROOT / 'results'
SOURCE_ZIP = INPUT_DIR / 'BrainTumor-Agent-source-multicase.zip'
MODEL_ZIP = INPUT_DIR / 'Dataset002_BRATS19.zip'
VALIDATION_ZIP = INPUT_DIR / 'BrainTumor-Agent-UPENN-10-independent-validation-cases.zip'
WORK_DIR = Path('/content/upenn-independent-validation')
PROJECT_DIR = WORK_DIR / 'BrainTumor-Agent'
NNUNET_ROOT = WORK_DIR / 'nnunet'
CASE_ROOT = WORK_DIR / 'test-data' / 'validation-cases'
OUTPUT_DIR = WORK_DIR / 'upenn-10-five-fold-output'

missing = [str(path) for path in (SOURCE_ZIP, MODEL_ZIP, VALIDATION_ZIP) if not path.is_file()]
if missing:
    raise FileNotFoundError('Upload the missing files first:\\n' + '\\n'.join(missing))
DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)
print('Input files found.')
"""
        ),
        code(
            """import subprocess
import sys

import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA was not detected. Select a T4 GPU runtime and run again.')
gpu_name = torch.cuda.get_device_name(0)
print('Torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', gpu_name)
subprocess.run(['nvidia-smi'], check=True)
"""
        ),
        code(
            """subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'nnunetv2==2.8.1', 'SimpleITK>=2.4,<3', 'nibabel>=5.3,<6'
], check=True)
print('Inference dependencies installed.')
"""
        ),
        code(
            """import os
import shutil
import zipfile

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True)
with zipfile.ZipFile(SOURCE_ZIP) as archive:
    archive.extractall(WORK_DIR)
with zipfile.ZipFile(VALIDATION_ZIP) as archive:
    archive.extractall(WORK_DIR / 'test-data')
results_root = NNUNET_ROOT / 'nnUNet_results'
results_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(MODEL_ZIP) as archive:
    archive.extractall(results_root)

os.environ['nnUNet_raw'] = str(NNUNET_ROOT / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(NNUNET_ROOT / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(results_root)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

case_dirs = sorted(path for path in CASE_ROOT.iterdir() if path.is_dir())
if len(case_dirs) != 10:
    raise RuntimeError(f'Expected 10 UPENN cases, found {len(case_dirs)}')
for case_dir in case_dirs:
    required = [case_dir / f'{case_dir.name}_{suffix}.nii.gz' for suffix in ('t1', 't1ce', 't2', 'flair', 'seg')]
    missing_case_files = [str(path) for path in required if not path.is_file()]
    if missing_case_files:
        raise FileNotFoundError('\\n'.join(missing_case_files))
case_ids = [path.name for path in case_dirs]
print('Cases:', ', '.join(case_ids))
"""
        ),
        code(
            """model_dir = (
    NNUNET_ROOT / 'nnUNet_results' / 'Dataset002_BRATS19' /
    'nnUNetTrainer__nnUNetPlans__3d_fullres'
)
checkpoints = [model_dir / f'fold_{fold}' / 'checkpoint_final.pth' for fold in range(5)]
missing_checkpoints = [str(path) for path in checkpoints if not path.is_file()]
if missing_checkpoints:
    raise FileNotFoundError('Five-fold weights are incomplete:\\n' + '\\n'.join(missing_checkpoints))
print('Five-fold weights found.')
"""
        ),
        code(
            """command = [
    sys.executable, '-m', 'segmentation.inference',
    '--input-dir', str(CASE_ROOT),
    '--output-dir', str(OUTPUT_DIR),
    '--nnunet-root', str(NNUNET_ROOT),
    '--dataset-id', '2',
    '--configuration', '3d_fullres',
    '--plans', 'nnUNetPlans',
    '--trainer', 'nnUNetTrainer',
    '--folds', '0', '1', '2', '3', '4',
    '--checkpoint', 'checkpoint_final.pth',
    '--device', 'cuda',
    '--gpu-id', '0',
    '--output-label-profile', 'brats19_preserved',
    '--preprocessing-processes', '2',
    '--export-processes', '2'
]
print('Starting 10-case five-fold GPU inference...')
subprocess.run(command, cwd=PROJECT_DIR, check=True)
"""
        ),
        code(
            """import json
from datetime import UTC, datetime

prediction_dir = OUTPUT_DIR / 'brats_predictions'
evaluation_json = OUTPUT_DIR / 'evaluation.json'
evaluation_csv = OUTPUT_DIR / 'evaluation_cases.csv'
evaluate_command = [
    sys.executable, '-m', 'segmentation.evaluate',
    '--ground-truth-dir', str(CASE_ROOT),
    '--prediction-dir', str(prediction_dir),
    '--ground-truth-label-space', 'brats',
    '--prediction-label-space', 'brats',
    '--output-json', str(evaluation_json),
    '--output-csv', str(evaluation_csv)
]
subprocess.run(evaluate_command, cwd=PROJECT_DIR, check=True)
payload = json.loads(evaluation_json.read_text(encoding='utf-8'))
payload['metadata'].update({
    'inference_folds': '0,1,2,3,4',
    'device': 'cuda',
    'gpu_name': gpu_name,
    'torch_version': torch.__version__,
    'nnunet_version': '2.8.1',
    'output_label_profile': 'brats19_preserved',
    'validation_cohort': ','.join(case_ids),
    'validation_source': 'UPENN-GBM expert-reviewed masks (TCIA, CC BY 4.0)',
    'validation_doi': '10.7937/TCIA.709X-DN49',
    'validation_caveat': 'Independent institutional cohort; 10-case sample, not a full clinical validation.',
    'validated_at': datetime.now(UTC).isoformat()
})
evaluation_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
payload
"""
        ),
        code(
            """archive_base = WORK_DIR / 'BrainTumor-Agent-UPENN-10-five-fold-GPU-validation'
archive_path = Path(shutil.make_archive(str(archive_base), 'zip', OUTPUT_DIR))
destination = DRIVE_RESULTS / archive_path.name
shutil.copy2(archive_path, destination)
shutil.copy2(evaluation_json, DRIVE_RESULTS / 'evaluation-upenn-10-cases.json')
shutil.copy2(evaluation_csv, DRIVE_RESULTS / 'evaluation-upenn-10-cases.csv')
summary = payload['summary']
print('UPENN-GBM 10-case five-fold GPU validation complete.')
for case in payload['cases']:
    print(f"{case['case_id']}: WT={case['whole_tumor']:.6f}, TC={case['tumor_core']:.6f}, ET={case['enhancing_tumor']:.6f}")
print(f"Macro WT Dice: {summary['whole_tumor']:.6f}")
print(f"Macro TC Dice: {summary['tumor_core']:.6f}")
print(f"Macro ET Dice: {summary['enhancing_tumor']:.6f}")
print('Result archive:', destination)
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    serialized = json.dumps(notebook, indent=1) + "\n"
    if OUTPUT.exists():
        existing = OUTPUT.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(f"Existing notebook differs from generated content: {OUTPUT}")
        print(f"Validated existing notebook: {OUTPUT}")
    else:
        OUTPUT.write_text(serialized, encoding="utf-8")
        print(f"Generated notebook: {OUTPUT}")
    print(f"Validated {sum(cell['cell_type'] == 'code' for cell in cells)} code cells")


if __name__ == "__main__":
    main()
