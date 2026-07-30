"""BraTS 2021病例目录和四模态文件发现。"""

from pathlib import Path

from data_process.constants import REQUIRED_MODALITIES, MRIModality
from data_process.exceptions import DatasetDiscoveryError
from data_process.io import is_nifti_path
from data_process.schemas import MRISeries, StudyManifest


def _match_modality(path: Path) -> MRIModality | None:
    """根据BraTS文件名后缀识别模态。"""

    lower_name = path.name.lower()
    for modality in REQUIRED_MODALITIES:
        if lower_name.endswith(f"_{modality.value}.nii.gz") or lower_name.endswith(
            f"_{modality.value}.nii"
        ):
            return modality
    return None


def _infer_case_id(path: Path, modality: MRIModality) -> str:
    """从“病例ID_模态.nii.gz”文件名中提取病例ID。"""

    lower_name = path.name.lower()
    suffixes = (f"_{modality.value}.nii.gz", f"_{modality.value}.nii")
    for suffix in suffixes:
        if lower_name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise DatasetDiscoveryError(f"无法从文件名推断病例ID：{path.name}")


def discover_study(case_dir: str | Path, case_id: str | None = None) -> StudyManifest:
    """发现一个BraTS病例目录中的T1、T1ce、T2和FLAIR文件。"""

    directory = Path(case_dir).expanduser().resolve()
    if not directory.is_dir():
        raise DatasetDiscoveryError(f"病例目录不存在：{directory}")

    discovered: dict[MRIModality, list[Path]] = {
        modality: [] for modality in REQUIRED_MODALITIES
    }
    for path in directory.iterdir():
        if not path.is_file() or not is_nifti_path(path):
            continue
        modality = _match_modality(path)
        if modality is not None:
            discovered[modality].append(path)

    errors: list[str] = []
    for modality, paths in discovered.items():
        if not paths:
            errors.append(f"缺少{modality.value}模态")
        elif len(paths) > 1:
            names = ", ".join(sorted(path.name for path in paths))
            errors.append(f"{modality.value}模态存在重复文件：{names}")
    if errors:
        raise DatasetDiscoveryError("；".join(errors))

    inferred_ids = {
        _infer_case_id(paths[0], modality)
        for modality, paths in discovered.items()
        if paths
    }
    if len(inferred_ids) != 1:
        raise DatasetDiscoveryError(
            f"四模态文件的病例ID不一致：{', '.join(sorted(inferred_ids))}"
        )

    resolved_case_id = case_id or inferred_ids.pop()
    series = tuple(
        MRISeries(modality=modality, path=discovered[modality][0])
        for modality in REQUIRED_MODALITIES
    )
    return StudyManifest(case_id=resolved_case_id, series=series)
