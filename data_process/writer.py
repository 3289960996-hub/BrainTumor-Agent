"""processed MRI文件与元数据保存。"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from data_process.constants import NNUNET_CHANNELS, REQUIRED_MODALITIES, MRIModality
from data_process.exceptions import ProcessedDataExistsError
from data_process.io import save_nifti
from data_process.normalization import NormalizationConfig
from data_process.schemas import LoadedStudy, ProcessedStudy


def _ensure_targets_available(targets: list[Path], overwrite: bool) -> None:
    """覆盖未授权时，避免破坏已经生成的processed数据。"""

    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise ProcessedDataExistsError(f"目标文件已存在：{names}")


def save_processed_study(
    study: LoadedStudy,
    normalized_data: NDArray[np.floating],
    output_root: str | Path,
    normalization_config: NormalizationConfig,
    overwrite: bool = False,
) -> ProcessedStudy:
    """保存四个归一化NIfTI及处理元数据。

    输出文件采用nnU-Net通道命名：
    T1=0000、T1ce=0001、T2=0002、FLAIR=0003。
    """

    data = np.asarray(normalized_data, dtype=np.float32)
    expected_shape = (len(REQUIRED_MODALITIES), *study.reference_geometry.shape)
    if data.shape != expected_shape:
        raise ValueError(f"processed数据shape应为{expected_shape}，实际为{data.shape}")

    case_output_dir = Path(output_root).expanduser().resolve() / study.case_id
    modality_files: dict[MRIModality, Path] = {
        modality: case_output_dir / f"{study.case_id}_{NNUNET_CHANNELS[modality]}.nii.gz"
        for modality in REQUIRED_MODALITIES
    }
    metadata_path = case_output_dir / "metadata.json"
    _ensure_targets_available([*modality_files.values(), metadata_path], overwrite)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    for index, modality in enumerate(REQUIRED_MODALITIES):
        save_nifti(
            data=data[index],
            reference=study.volumes[modality],
            output_path=modality_files[modality],
        )

    geometry = study.reference_geometry
    metadata = {
        "format_version": "1.0",
        "case_id": study.case_id,
        "created_at": datetime.now(UTC).isoformat(),
        "channel_order": [
            {
                "index": index,
                "channel": NNUNET_CHANNELS[modality],
                "modality": modality.value,
            }
            for index, modality in enumerate(REQUIRED_MODALITIES)
        ],
        "source_files": {
            modality.value: str(study.volumes[modality].path)
            for modality in REQUIRED_MODALITIES
        },
        "processed_files": {
            modality.value: modality_files[modality].name
            for modality in REQUIRED_MODALITIES
        },
        "shape": list(geometry.shape),
        "spacing": list(geometry.spacing),
        "origin": list(geometry.origin),
        "direction": list(geometry.direction),
        "affine": geometry.affine.tolist(),
        "normalization": {
            "method": "numpy.channel_wise_zscore",
            "nonzero": normalization_config.nonzero,
            "channel_wise": normalization_config.channel_wise,
            "dtype": "float32",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return ProcessedStudy(
        case_id=study.case_id,
        output_dir=case_output_dir,
        modality_files=modality_files,
        metadata_path=metadata_path,
    )
