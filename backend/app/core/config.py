"""Environment-based application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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
    max_upload_size_mb: int = Field(default=1024, ge=1, le=4096)

    database_url: str = "sqlite:///./runtime/brain_tumor_agent.db"
    redis_url: str = "redis://127.0.0.1:6379/0"
    celery_task_always_eager: bool = False
    analysis_task_max_retries: int = Field(default=1, ge=0, le=5)
    analysis_task_retry_delay_seconds: int = Field(default=30, ge=0, le=3600)

    nnunet_root: Path = Field(
        default=Path("./runtime/nnunet"),
        validation_alias="NNUNET_ROOT",
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


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
