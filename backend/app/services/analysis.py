"""MRI预处理、nnU-Net推理与肿瘤量化流水线。"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from backend.app.services.errors import CaseStateError, PipelineExecutionError
from backend.app.services.storage import CasePaths, CaseRepository
from data_process.constants import NNUNET_CHANNELS, REQUIRED_MODALITIES
from data_process.processor import BraTSDataProcessor
from feature_extract.tumor_measure import TumorMeasurement, measure_tumor
from segmentation.inference import build_predict_command, prepare_inference_inputs
from segmentation.prepare_dataset import (
    DEFAULT_CONFIGURATION,
    DEFAULT_DATASET_ID,
    DEFAULT_PLANS,
    OutputLabelProfile,
    SegmentationSetupError,
    configure_nnunet_environment,
    restore_segmentation_file,
    run_command,
)
from segmentation.train import configure_device, normalize_folds

ANALYSIS_CACHE_VERSION = 2


class AnalysisCancellationRequested(Exception):
    """Raised by a task progress hook at a safe pipeline stage boundary."""


class PreprocessorProtocol(Protocol):
    def prepare(self, paths: CasePaths) -> Path:
        """返回nnU-Net四通道输入目录。"""


class PredictorProtocol(Protocol):
    def predict(self, input_dir: Path, output_root: Path, case_id: str) -> Path:
        """返回BraTS标签空间的mask。"""


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    case_id: str
    mask_path: Path
    metrics: dict[str, Any]


class MRIProcessingService:
    """复用data_process模块并支持安全续跑。"""

    def __init__(self, processor: BraTSDataProcessor | None = None) -> None:
        self.processor = processor or BraTSDataProcessor()

    def prepare(self, paths: CasePaths) -> Path:
        expected = [
            paths.processed_case
            / f"{paths.case_id}_{NNUNET_CHANNELS[modality]}.nii.gz"
            for modality in REQUIRED_MODALITIES
        ]
        if all(path.is_file() for path in expected):
            return paths.processed_case

        # 病例目录由本服务生成；若上一次处理部分失败，仅覆盖该病例的processed产物。
        processed = self.processor.process_case(
            case_dir=paths.raw,
            output_root=paths.processed_root,
            case_id=paths.case_id,
            overwrite=paths.processed_case.exists(),
        )
        return processed.output_dir


@dataclass(frozen=True, slots=True)
class NNUNetInferenceConfig:
    nnunet_root: Path
    dataset_id: int = DEFAULT_DATASET_ID
    configuration: str = DEFAULT_CONFIGURATION
    plans: str = DEFAULT_PLANS
    trainer: str = "nnUNetTrainer"
    folds: tuple[str, ...] = ("0", "1", "2", "3", "4")
    device: str = "cuda"
    gpu_id: str = "0"
    checkpoint: str = "checkpoint_final.pth"
    step_size: float = 0.5
    preprocessing_processes: int = 3
    export_processes: int = 3
    disable_tta: bool = False
    output_label_profile: OutputLabelProfile = "standard_nnunet"


class NNUNetInferenceService:
    """调用已训练的nnU-Net V2并把内部标签还原为BraTS标签。"""

    def __init__(self, config: NNUNetInferenceConfig) -> None:
        self.config = config
        # 单进程内串行使用GPU，避免多个病例并行推理造成显存竞争。
        self._inference_lock = threading.Lock()

    def predict(self, input_dir: Path, output_root: Path, case_id: str) -> Path:
        with self._inference_lock:
            return self._predict_locked(input_dir, output_root, case_id)

    def _predict_locked(
        self,
        input_dir: Path,
        output_root: Path,
        case_id: str,
    ) -> Path:
        brats_output = output_root / "brats_predictions"
        restored_mask = brats_output / f"{case_id}.nii.gz"
        nnunet_output = output_root / "nnunet_predictions"
        internal_mask = nnunet_output / f"{case_id}.nii.gz"
        if internal_mask.is_file():
            return restore_segmentation_file(
                internal_mask,
                restored_mask,
                output_label_profile=self.config.output_label_profile,
            )
        if restored_mask.is_file():
            return restored_mask

        configure_nnunet_environment(self.config.nnunet_root)
        folds = normalize_folds(self.config.folds)
        configure_device(
            device=self.config.device,
            gpu_ids=self.config.gpu_id,
            num_gpus=1,
            dry_run=False,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        standardized_input, case_ids = prepare_inference_inputs(
            input_dir=input_dir,
            staging_dir=output_root / "nnunet_input",
        )
        if case_id not in case_ids:
            raise SegmentationSetupError(
                f"推理输入中未找到病例{case_id}：{case_ids}"
            )
        preprocessing_processes = self.config.preprocessing_processes
        export_processes = self.config.export_processes
        if self.config.device == "cpu":
            preprocessing_processes = 1
            export_processes = 1
        command = build_predict_command(
            input_dir=standardized_input,
            output_dir=nnunet_output,
            dataset_id=self.config.dataset_id,
            configuration=self.config.configuration,
            plans=self.config.plans,
            trainer=self.config.trainer,
            folds=folds,
            device=self.config.device,
            checkpoint=self.config.checkpoint,
            step_size=self.config.step_size,
            preprocessing_processes=preprocessing_processes,
            export_processes=export_processes,
            disable_tta=self.config.disable_tta,
        )
        run_command(command)
        if not internal_mask.is_file():
            raise SegmentationSetupError(
                f"nnU-Net未生成预期mask：{internal_mask.name}"
            )
        return restore_segmentation_file(
            internal_mask,
            restored_mask,
            output_label_profile=self.config.output_label_profile,
        )


class MRIAnalysisPipeline:
    """按预处理→分割→量化顺序执行单病例分析。"""

    def __init__(
        self,
        repository: CaseRepository,
        preprocessor: PreprocessorProtocol,
        predictor: PredictorProtocol,
    ) -> None:
        self.repository = repository
        self.preprocessor = preprocessor
        self.predictor = predictor

    def analyze(
        self,
        case_id: str,
        progress: Callable[[str, int, str], None] | None = None,
    ) -> AnalysisResult:
        report_progress = progress or (lambda _stage, _percent, _message: None)
        with self.repository.case_lock(case_id):
            paths = self.repository.require_case(case_id)
            cached = self.repository.load_features(case_id)
            status_payload = self.repository.read_status(case_id)
            if (
                paths.mask.is_file()
                and cached is not None
                and status_payload.get("analysis_cache_version")
                == ANALYSIS_CACHE_VERSION
            ):
                return AnalysisResult(
                    case_id=paths.case_id,
                    mask_path=paths.mask,
                    metrics=cached,
                )

            status = str(status_payload.get("status", ""))
            if status not in {
                "uploaded",
                "analysis_failed",
                "analyzed",
                "report_ready",
            }:
                raise CaseStateError(
                    f"病例{case_id}当前状态为{status or 'unknown'}，不能执行分析"
                )

            self.repository.write_status(case_id, "analyzing")
            try:
                report_progress("preprocessing", 10, "正在校验并预处理四模态MRI")
                processed_dir = self.preprocessor.prepare(paths)
                report_progress("inference", 30, "正在执行nnU-Net肿瘤分割")
                mask_path = self.predictor.predict(
                    input_dir=processed_dir,
                    output_root=paths.inference_root,
                    case_id=paths.case_id,
                )
                report_progress("measurement", 85, "正在计算WT、TC和ET定量指标")
                measurement: TumorMeasurement = measure_tumor(
                    mask_path,
                    label_space="brats",
                )
                metrics = measurement.to_dict()
                report_progress("saving", 95, "正在保存分析结果")
                self.repository.save_features(case_id, metrics)
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.CalledProcessError,
            ) as exc:
                self.repository.write_status(
                    case_id,
                    "analysis_failed",
                    extra={"failure_type": type(exc).__name__},
                )
                if isinstance(exc, subprocess.CalledProcessError) and exc.returncode in {
                    1,
                    3221225477,
                    -1073741819,
                }:
                    message = (
                        "nnU-Net推理失败：系统可用内存不足。已启用CPU低内存模式，"
                        "请关闭占用内存较大的程序后重试"
                    )
                else:
                    message = "MRI分析执行失败，请检查四模态数据、nnU-Net模型和设备配置"
                raise PipelineExecutionError(message) from exc

            self.repository.write_status(
                case_id,
                "analyzed",
                extra={
                    "mask_filename": mask_path.name,
                    "features_filename": paths.features.name,
                    "analysis_cache_version": ANALYSIS_CACHE_VERSION,
                    "label_mapping_revision": 2,
                    "report_stale": paths.report.is_file(),
                },
            )
            return AnalysisResult(
                case_id=paths.case_id,
                mask_path=mask_path,
                metrics=metrics,
            )
