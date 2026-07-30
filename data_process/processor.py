"""BraTS 2021单病例数据处理流程编排。"""

from pathlib import Path

from data_process.loader import load_brats_case
from data_process.normalization import NormalizationConfig, normalize_multimodal
from data_process.schemas import ProcessedStudy
from data_process.writer import save_processed_study


class BraTSDataProcessor:
    """依次执行发现、读取、检查、归一化和保存。"""

    def __init__(
        self,
        normalization_config: NormalizationConfig | None = None,
        spacing_tolerance: float = 1e-5,
        affine_tolerance: float = 1e-4,
    ) -> None:
        self.normalization_config = normalization_config or NormalizationConfig()
        self.spacing_tolerance = spacing_tolerance
        self.affine_tolerance = affine_tolerance

    def process_case(
        self,
        case_dir: str | Path,
        output_root: str | Path,
        case_id: str | None = None,
        overwrite: bool = False,
    ) -> ProcessedStudy:
        """处理一个BraTS病例并保存nnU-Net兼容的四模态NIfTI。"""

        study = load_brats_case(
            case_dir=case_dir,
            case_id=case_id,
            spacing_tolerance=self.spacing_tolerance,
            affine_tolerance=self.affine_tolerance,
        )
        normalized = normalize_multimodal(
            study.stacked_data(),
            config=self.normalization_config,
        )
        return save_processed_study(
            study=study,
            normalized_data=normalized,
            output_root=output_root,
            normalization_config=self.normalization_config,
            overwrite=overwrite,
        )
