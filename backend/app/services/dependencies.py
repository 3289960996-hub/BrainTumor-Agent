"""FastAPI依赖工厂及生产环境默认实现。"""

from functools import lru_cache

from agent.qwen import QwenAgentConfig
from backend.app.core.config import get_settings
from backend.app.services.analysis import (
    MRIAnalysisPipeline,
    MRIProcessingService,
    NNUNetInferenceConfig,
    NNUNetInferenceService,
)
from backend.app.services.chat import MedicalAgentChatService
from backend.app.services.reporting import MedicalReportService
from backend.app.services.storage import CaseRepository
from backend.app.services.upload import MRIUploadService
from report.generator import ReportConfig


def _api_key() -> str:
    secret = get_settings().dashscope_api_key
    return secret.get_secret_value() if secret is not None else ""


@lru_cache
def get_case_repository() -> CaseRepository:
    settings = get_settings()
    return CaseRepository(settings.data_root)


@lru_cache
def get_upload_service() -> MRIUploadService:
    settings = get_settings()
    return MRIUploadService(
        get_case_repository(),
        max_file_bytes=settings.max_upload_size_mb * 1024 * 1024,
    )


@lru_cache
def get_analysis_pipeline() -> MRIAnalysisPipeline:
    settings = get_settings()
    predictor = NNUNetInferenceService(
        NNUNetInferenceConfig(
            nnunet_root=settings.nnunet_root,
            dataset_id=settings.nnunet_dataset_id,
            configuration=settings.nnunet_configuration,
            plans=settings.nnunet_plans,
            trainer=settings.nnunet_trainer,
            folds=tuple(settings.nnunet_folds),
            device=settings.nnunet_device,
            gpu_id=settings.nnunet_gpu_id,
            checkpoint=settings.nnunet_checkpoint,
            output_label_profile=settings.nnunet_output_label_profile,
            step_size=settings.nnunet_step_size,
            preprocessing_processes=settings.nnunet_preprocessing_processes,
            export_processes=settings.nnunet_export_processes,
            disable_tta=settings.nnunet_disable_tta,
        )
    )
    return MRIAnalysisPipeline(
        repository=get_case_repository(),
        preprocessor=MRIProcessingService(),
        predictor=predictor,
    )


@lru_cache
def get_report_service() -> MedicalReportService:
    settings = get_settings()
    return MedicalReportService(
        get_case_repository(),
        config=ReportConfig(
            api_key=_api_key(),
            base_url=settings.qwen_base_url,
            model=settings.qwen_model,
            temperature=settings.report_temperature,
            max_tokens=settings.report_max_tokens,
            timeout_seconds=settings.qwen_timeout_seconds,
        ),
    )


@lru_cache
def get_chat_service() -> MedicalAgentChatService:
    settings = get_settings()
    return MedicalAgentChatService(
        repository=get_case_repository(),
        rag_index_path=settings.faiss_index_path,
        config=QwenAgentConfig(
            api_key=_api_key(),
            base_url=settings.qwen_base_url,
            model=settings.qwen_model,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
            timeout_seconds=settings.qwen_timeout_seconds,
        ),
    )


def clear_service_caches() -> None:
    """测试或配置热加载时清理进程内单例。"""

    get_upload_service.cache_clear()
    get_analysis_pipeline.cache_clear()
    get_report_service.cache_clear()
    get_chat_service.cache_clear()
    get_case_repository.cache_clear()
