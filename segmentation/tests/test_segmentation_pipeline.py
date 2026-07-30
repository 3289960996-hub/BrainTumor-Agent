"""使用合成NIfTI测试nnU-Net数据转换、命令和Dice评估。"""

import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from segmentation.evaluate import dice_score, evaluate_folders, labels_to_regions
from segmentation.inference import (
    build_predict_command,
    prepare_inference_inputs,
    restore_prediction_folder,
)
from segmentation.prepare_dataset import (
    DEFAULT_PLANS,
    convert_brats19_preserved_to_brats_labels,
    convert_brats_to_nnunet_labels,
    convert_nnunet_to_brats_labels,
    prepare_brats_dataset,
)
from segmentation.train import build_train_command, normalize_folds


def _write_brats_case(
    root: Path,
    case_id: str = "BraTS2021_00010",
) -> Path:
    """创建一个包含四模态和0/1/2/4标签的合成BraTS病例。"""

    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    shape = (8, 9, 10)
    affine = np.eye(4, dtype=np.float64)

    for index, modality in enumerate(("t1", "t1ce", "t2", "flair")):
        data = np.indices(shape, dtype=np.float32).sum(axis=0) + index * 10.0
        nib.save(
            nib.Nifti1Image(data.astype(np.float32), affine),
            case_dir / f"{case_id}_{modality}.nii.gz",
        )

    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[1:7, 1:8, 2:8] = 2
    segmentation[2:6, 2:7, 3:7] = 1
    segmentation[3:5, 3:6, 4:6] = 4
    nib.save(
        nib.Nifti1Image(segmentation, affine),
        case_dir / f"{case_id}_seg.nii.gz",
    )
    return case_dir


def test_label_conversion_round_trip() -> None:
    """BraTS和nnU-Net标签映射必须可逆。"""

    brats = np.asarray([0, 1, 2, 4], dtype=np.uint8)
    nnunet = convert_brats_to_nnunet_labels(brats)

    assert nnunet.tolist() == [0, 2, 1, 3]
    assert convert_nnunet_to_brats_labels(nnunet).tolist() == brats.tolist()


def test_brats19_five_class_output_conversion() -> None:
    """Dataset002自定义五类输出应还原到标准BraTS标签。"""

    model_output = np.asarray([0, 1, 2, 3, 4], dtype=np.uint8)

    restored = convert_brats19_preserved_to_brats_labels(model_output)

    assert restored.tolist() == [0, 2, 1, 0, 4]


def test_prepare_dataset_generates_channels_labels_and_json(tmp_path: Path) -> None:
    """转换脚本应生成四通道、连续标签和区域训练dataset.json。"""

    brats_root = tmp_path / "BraTS2021"
    nnunet_root = tmp_path / "nnunet"
    case_id = "BraTS2021_00010"
    _write_brats_case(brats_root, case_id)

    result = prepare_brats_dataset(
        brats_root=brats_root,
        nnunet_root=nnunet_root,
        dataset_id=137,
        dataset_name="BraTS2021",
        workers=1,
    )

    assert result.num_training_cases == 1
    assert Path(os.environ["nnUNet_raw"]) == nnunet_root / "nnUNet_raw"
    for channel in ("0000", "0001", "0002", "0003"):
        assert (result.dataset_dir / "imagesTr" / f"{case_id}_{channel}.nii.gz").is_file()

    converted_path = result.dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
    converted = np.asarray(nib.load(converted_path).dataobj)
    assert set(int(value) for value in np.unique(converted)) == {0, 1, 2, 3}

    dataset_json = json.loads(result.dataset_json_path.read_text(encoding="utf-8"))
    assert dataset_json["channel_names"] == {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR",
    }
    assert dataset_json["labels"]["whole tumor"] == [1, 2, 3]
    assert dataset_json["labels"]["tumor core"] == [2, 3]
    assert dataset_json["labels"]["enhancing tumor"] == [3]
    assert dataset_json["regions_class_order"] == [1, 2, 3]
    assert dataset_json["numTraining"] == 1


def test_train_and_predict_commands_are_gpu_ready(tmp_path: Path) -> None:
    """命令应携带3D配置、ResEnc计划、fold和CUDA参数。"""

    train_command = build_train_command(
        dataset_id=137,
        configuration="3d_fullres",
        fold="0",
        plans=DEFAULT_PLANS,
        device="cuda",
        num_gpus=1,
        save_npz=True,
    )
    assert train_command[:4] == ["nnUNetv2_train", "137", "3d_fullres", "0"]
    assert DEFAULT_PLANS in train_command
    assert "--npz" in train_command
    assert train_command[train_command.index("-device") + 1] == "cuda"

    predict_command = build_predict_command(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        dataset_id=137,
        configuration="3d_fullres",
        plans=DEFAULT_PLANS,
        trainer="nnUNetTrainer",
        folds=("0", "1", "2", "3", "4"),
        device="cuda",
        checkpoint="checkpoint_final.pth",
        step_size=0.5,
        preprocessing_processes=2,
        export_processes=2,
    )
    assert predict_command[0] == "nnUNetv2_predict"
    assert predict_command[predict_command.index("-f") + 1 : predict_command.index("-device")] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert normalize_folds(("0", "1", "1")) == ("0", "1")


def test_inference_staging_and_label_restore(tmp_path: Path) -> None:
    """原始BraTS输入应转换为推理通道，预测可还原为BraTS标签。"""

    brats_root = tmp_path / "BraTS2021"
    case_id = "BraTS2021_00010"
    case_dir = _write_brats_case(brats_root, case_id)
    staging, case_ids = prepare_inference_inputs(
        input_dir=brats_root,
        staging_dir=tmp_path / "staging",
    )

    assert case_ids == (case_id,)
    assert all(
        (staging / f"{case_id}_{channel}.nii.gz").is_file()
        for channel in ("0000", "0001", "0002", "0003")
    )

    internal_prediction_dir = tmp_path / "nnunet_predictions"
    internal_prediction_dir.mkdir()
    brats_segmentation = np.asarray(
        nib.load(case_dir / f"{case_id}_seg.nii.gz").dataobj
    )
    internal = convert_brats_to_nnunet_labels(brats_segmentation)
    nib.save(
        nib.Nifti1Image(internal, np.eye(4)),
        internal_prediction_dir / f"{case_id}.nii.gz",
    )

    restored_paths = restore_prediction_folder(
        internal_prediction_dir,
        tmp_path / "brats_predictions",
    )
    restored = np.asarray(nib.load(restored_paths[0]).dataobj)
    assert np.array_equal(restored, brats_segmentation)


def test_brats19_prediction_folder_restore(tmp_path: Path) -> None:
    """五类BraTS19预测目录应使用专用标签配置还原。"""

    internal_prediction_dir = tmp_path / "nnunet_predictions"
    internal_prediction_dir.mkdir()
    internal = np.asarray([0, 1, 2, 3, 4], dtype=np.uint8).reshape(1, 1, 5)
    nib.save(
        nib.Nifti1Image(internal, np.eye(4)),
        internal_prediction_dir / "case.nii.gz",
    )

    restored_paths = restore_prediction_folder(
        internal_prediction_dir,
        tmp_path / "brats_predictions",
        output_label_profile="brats19_preserved",
    )
    restored = np.asarray(nib.load(restored_paths[0]).dataobj)

    assert restored.reshape(-1).tolist() == [0, 2, 1, 0, 4]


def test_evaluate_outputs_perfect_region_dice(tmp_path: Path) -> None:
    """预测与真值一致时WT、TC和ET Dice都应为1。"""

    ground_truth_root = tmp_path / "ground_truth"
    prediction_dir = tmp_path / "predictions"
    case_id = "BraTS2021_00010"
    case_dir = _write_brats_case(ground_truth_root, case_id)
    prediction_dir.mkdir()
    shutil.copy2(
        case_dir / f"{case_id}_seg.nii.gz",
        prediction_dir / f"{case_id}.nii.gz",
    )

    summary, cases = evaluate_folders(
        ground_truth_dir=ground_truth_root,
        prediction_dir=prediction_dir,
    )

    assert summary.num_cases == 1
    assert summary.whole_tumor == pytest.approx(1.0)
    assert summary.tumor_core == pytest.approx(1.0)
    assert summary.enhancing_tumor == pytest.approx(1.0)
    assert cases[0].case_id == case_id


def test_region_dice_handles_empty_et() -> None:
    """预测和真值ET都为空时，ET Dice按约定为1。"""

    labels = np.asarray([0, 1, 2], dtype=np.uint8)
    regions = labels_to_regions(labels, "brats")

    assert dice_score(regions["enhancing_tumor"], regions["enhancing_tumor"]) == 1.0
