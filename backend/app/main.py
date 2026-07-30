"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.router import api_router
from backend.app.core.config import get_settings
from backend.app.services.errors import BackendServiceError

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description="Backend API for multimodal MRI brain tumor assisted analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
