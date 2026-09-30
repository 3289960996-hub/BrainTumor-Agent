"""Environment-based application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration.

    Non-secret defaults may also be documented in configs/app.yaml. Secrets must
    only be supplied through environment variables or a secret manager.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BTA_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "BrainTumor-Agent"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    data_root: Path = Path("./runtime/data")
    model_root: Path = Path("./runtime/models")
    faiss_index_path: Path = Path("./runtime/faiss")
    # 本地医学RAG embedding：只在这里定义一次，rag/embedding.py 通过Settings读取，
    # 因此写在.env里的值（而不是仅导出到shell的变量）也会生效。
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    embedding_batch_size: int = Field(default=16, ge=1, le=512)
    embedding_cache_dir: Path | None = None
    embedding_local_files_only: bool = False
    max_upload_size_mb: int = Field(default=1024, ge=1, le=4096)
    # 四个模态合计上限：单文件上限乘以模态数会放大磁盘耗尽风险。
    max_upload_total_mb: int = Field(default=2048, ge=1, le=16384)
    # NIfTI头声明的体素数上限：极小文件可以声明巨大的解压后体积。
    max_nifti_voxels: int = Field(default=300_000_000, ge=1_000_000)
    # 每个队列同时允许的活动任务数，超出返回429。
    max_active_tasks: int = Field(default=2, ge=1, le=64)

    redis_url: str = "redis://127.0.0.1:6379/0"
    celery_task_always_eager: bool = False
    analysis_task_max_retries: int = Field(default=1, ge=0, le=5)
    analysis_task_retry_delay_seconds: int = Field(default=30, ge=0, le=3600)

    nnunet_root: Path = Field(
        default=Path("./runtime/nnunet"),
        validation_alias=AliasChoices("BTA_NNUNET_ROOT", "NNUNET_ROOT"),
    )
    nnunet_dataset_id: int = Field(default=137, ge=1, le=999)
    nnunet_configuration: str = "3d_fullres"
    nnunet_plans: str = "nnUNetResEncUNetMPlans"
    nnunet_trainer: str = "nnUNetTrainer"
    nnunet_folds: list[str] = ["0", "1", "2", "3", "4"]
    nnunet_device: Literal["cuda", "cpu", "mps"] = "cuda"
    nnunet_gpu_id: str = "0"
    nnunet_checkpoint: str = "checkpoint_final.pth"
    nnunet_output_label_profile: Literal[
        "standard_nnunet",
        "brats19_preserved",
    ] = "standard_nnunet"
    nnunet_step_size: float = Field(default=0.5, gt=0.0, le=1.0)
    nnunet_preprocessing_processes: int = Field(default=3, ge=1)
    nnunet_export_processes: int = Field(default=3, ge=1)
    nnunet_disable_tta: bool = False

    qwen_model: str = "qwen-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_timeout_seconds: float = Field(default=60.0, gt=0.0)
    qwen_enable_data_inspection: bool = False
    agent_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    agent_max_tokens: int = Field(default=1000, ge=100)
    report_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    report_max_tokens: int = Field(default=800, ge=100)
    dashscope_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="DASHSCOPE_API_KEY",
    )
    api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
