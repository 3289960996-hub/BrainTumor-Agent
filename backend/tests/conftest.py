"""Shared test authentication override for protected API routes."""

import pytest
from fastapi import Request

from backend.app.main import app
from backend.app.services.security import AuthContext, authenticate_request


@pytest.fixture(autouse=True)
def authenticated_test_client():
    def _authenticate(request: Request):
        context = AuthContext(principal="default-client")
        request.state.auth = context
        return context

    app.dependency_overrides[authenticate_request] = _authenticate
    yield
    app.dependency_overrides.pop(authenticate_request, None)
