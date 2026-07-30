"""Service health endpoint."""

from typing import Any

from fastapi import APIRouter

from backend.app.core.config import get_settings

router = APIRouter()


@router.get("/health", summary="Check API health")
async def health_check() -> dict[str, Any]:
    """返回API状态及各可选能力的运行时就绪情况。"""

    settings = get_settings()
    secret = settings.dashscope_api_key
    qwen_ready = bool(secret and secret.get_secret_value().strip())
    results_root = settings.nnunet_root / "nnUNet_results"
    analysis_ready = results_root.is_dir() and any(
        results_root.rglob(settings.nnunet_checkpoint)
    )
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "capabilities": {
            "upload": True,
            "analysis": analysis_ready,
            "report": qwen_ready,
            "chat": qwen_ready,
            "rag": settings.faiss_index_path.is_dir(),
        },
    }
