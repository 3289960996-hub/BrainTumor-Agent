"""API authentication, case ownership checks, and request audit helpers."""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

from backend.app.core.config import get_settings
from backend.app.services.errors import AuthenticationError, CaseNotFoundError, ServiceConfigurationError

if TYPE_CHECKING:
    from backend.app.services.storage import CaseRepository

logger = logging.getLogger("brain_tumor_agent.audit")
DEFAULT_PRINCIPAL = "default-client"


@dataclass(frozen=True, slots=True)
class AuthContext:
    principal: str


def authenticate_request(request: Request) -> AuthContext:
    """Authenticate protected API requests with a configured API key."""

    configured = get_settings().api_key
    if configured is None or not configured.get_secret_value():
        raise ServiceConfigurationError("未配置BTA_API_KEY，受保护API已拒绝请求")
    supplied = request.headers.get("x-api-key", "")
    if not supplied:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
    if not supplied or not secrets.compare_digest(supplied, configured.get_secret_value()):
        raise AuthenticationError()
    context = AuthContext(principal=DEFAULT_PRINCIPAL)
    request.state.auth = context
    return context


def principal_for(request: Request) -> str:
    context = getattr(request.state, "auth", None)
    return context.principal if isinstance(context, AuthContext) else DEFAULT_PRINCIPAL


def require_case_access(repository: CaseRepository, case_id: str, request: Request) -> None:
    """Authorize a case while hiding whether another principal owns it."""

    paths = repository.require_case(case_id)
    payload = repository.read_status(paths.case_id)
    if payload.get("owner_id", DEFAULT_PRINCIPAL) != principal_for(request):
        raise CaseNotFoundError(paths.case_id)


def audit_event(event: str, request: Request, **fields: object) -> None:
    safe_fields = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    logger.info(
        "event=%s principal=%s method=%s path=%s %s",
        event,
        principal_for(request),
        request.method,
        request.url.path,
        safe_fields,
    )
