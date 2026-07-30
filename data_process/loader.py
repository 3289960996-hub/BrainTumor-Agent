"""BraTS四模态MRI加载器。"""

from pathlib import Path

from data_process.constants import REQUIRED_MODALITIES
from data_process.discovery import discover_study
from data_process.io import read_nifti
from data_process.schemas import LoadedStudy, StudyManifest
from data_process.validator import ensure_compatible_geometry, ensure_valid_manifest


def load_study_from_manifest(
    manifest: StudyManifest,
    spacing_tolerance: float = 1e-5,
    affine_tolerance: float = 1e-4,
) -> LoadedStudy:
    """从显式病例清单读取T1、T1ce、T2和FLAIR。"""

    ensure_valid_manifest(manifest)
    modality_map = manifest.by_modality()
    volumes = {
        modality: read_nifti(modality_map[modality].path)
        for modality in REQUIRED_MODALITIES
    }
    ensure_compatible_geometry(
        volumes,
        spacing_tolerance=spacing_tolerance,
        affine_tolerance=affine_tolerance,
    )
    return LoadedStudy(case_id=manifest.case_id, volumes=volumes)


def load_brats_case(
    case_dir: str | Path,
    case_id: str | None = None,
    spacing_tolerance: float = 1e-5,
    affine_tolerance: float = 1e-4,
) -> LoadedStudy:
    """从标准BraTS病例目录自动发现并加载四模态MRI。"""

    manifest = discover_study(case_dir=case_dir, case_id=case_id)
    return load_study_from_manifest(
        manifest,
        spacing_tolerance=spacing_tolerance,
        affine_tolerance=affine_tolerance,
    )
