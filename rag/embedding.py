"""BGE医学知识库Embedding配置与创建。"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

DEFAULT_BGE_MODEL = "BAAI/bge-m3"
EmbeddingFactory = Callable[..., Embeddings]


class EmbeddingSetupError(ValueError):
    """Embedding配置或运行环境不合法。"""


@dataclass(frozen=True, slots=True)
class BGEEmbeddingConfig:
    """本地BGE-M3 dense embedding配置。"""

    model_name: str = DEFAULT_BGE_MODEL
    device: str = "auto"
    batch_size: int = 16
    normalize_embeddings: bool = True
    cache_dir: Path | None = None
    local_files_only: bool = False

    @classmethod
    def from_env(cls) -> BGEEmbeddingConfig:
        """从环境变量读取Embedding设置。"""

        cache_value = os.getenv("BTA_EMBEDDING_CACHE_DIR", "").strip()
        return cls(
            model_name=os.getenv("BTA_EMBEDDING_MODEL", DEFAULT_BGE_MODEL).strip(),
            device=os.getenv("BTA_EMBEDDING_DEVICE", "auto").strip(),
            batch_size=int(os.getenv("BTA_EMBEDDING_BATCH_SIZE", "16")),
            cache_dir=Path(cache_value).expanduser().resolve() if cache_value else None,
            local_files_only=_env_bool("BTA_EMBEDDING_LOCAL_FILES_ONLY", False),
        )

    def validate(self) -> None:
        """验证模型和批量参数。"""

        if not self.model_name:
            raise EmbeddingSetupError("Embedding model_name不能为空")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise EmbeddingSetupError("device必须是auto、cpu、cuda或mps")
        if self.batch_size < 1:
            raise EmbeddingSetupError("batch_size必须大于等于1")


def resolve_device(requested: str) -> str:
    """选择可用计算设备；auto优先CUDA，其次MPS，最后CPU。"""

    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def create_bge_embeddings(
    config: BGEEmbeddingConfig | None = None,
    factory: EmbeddingFactory | None = None,
) -> Embeddings:
    """创建LangChain HuggingFaceEmbeddings。

    BGE-M3不需要额外query instruction；文档和查询向量均执行L2归一化。
    首次运行会从Hugging Face下载模型，除非启用local_files_only。
    """

    resolved_config = config or BGEEmbeddingConfig.from_env()
    resolved_config.validate()
    embedding_factory = factory or _default_factory()
    model_kwargs: dict[str, Any] = {
        "device": resolve_device(resolved_config.device),
        "local_files_only": resolved_config.local_files_only,
    }
    factory_kwargs: dict[str, Any] = {
        "model_name": resolved_config.model_name,
        "model_kwargs": model_kwargs,
        "encode_kwargs": {
            "normalize_embeddings": resolved_config.normalize_embeddings,
            "batch_size": resolved_config.batch_size,
            "show_progress_bar": False,
        },
    }
    if resolved_config.cache_dir is not None:
        factory_kwargs["cache_folder"] = str(resolved_config.cache_dir)
    try:
        return embedding_factory(**factory_kwargs)
    except Exception as exc:
        raise EmbeddingSetupError(
            f"无法加载Embedding模型{resolved_config.model_name}：{exc}"
        ) from exc


def _default_factory() -> EmbeddingFactory:
    """延迟导入独立的LangChain Hugging Face集成包。"""

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise EmbeddingSetupError(
            "缺少langchain-huggingface，请先安装requirements.txt"
        ) from exc
    return HuggingFaceEmbeddings


def _env_bool(name: str, default: bool) -> bool:
    """解析常见布尔环境变量。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise EmbeddingSetupError(f"{name}必须是布尔值")
