"""FastAPI application entry point."""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.router import api_router
from backend.app.core.config import get_settings
from backend.app.core.logging_setup import configure_logging
from backend.app.services.errors import BackendServiceError

configure_logging()
settings = get_settings()
audit_logger = logging.getLogger("brain_tumor_agent.audit")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description="Backend API for multimodal MRI brain tumor assisted analysis.",
    # production 环境默认关闭接口文档与 schema，减少不必要的公开面；
    # 可用 BTA_EXPOSE_API_DOCS 显式覆盖（见 Settings.api_docs_enabled）。
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_requests(request: Request, call_next):
    response = await call_next(request)
    context = getattr(request.state, "auth", None)
    principal = getattr(context, "principal", "anonymous")
    audit_logger.info(
        "event=api_request principal=%s method=%s path=%s status=%s",
        principal,
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.exception_handler(BackendServiceError)
async def backend_service_error_handler(
    request: Request,
    exc: BackendServiceError,
) -> JSONResponse:
    """将服务层错误转换为稳定且不泄漏堆栈的API响应。"""

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": {
                "code": exc.code,
                "message": exc.message,
                "path": request.url.path,
            }
        },
    )


app.include_router(api_router, prefix=settings.api_prefix)
