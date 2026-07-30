"""Top-level API router."""

from fastapi import APIRouter

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.imaging import router as imaging_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["system"])
api_router.include_router(imaging_router, tags=["mri-assistant"])
