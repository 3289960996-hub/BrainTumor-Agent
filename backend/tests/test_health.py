"""Smoke test for the initial API skeleton."""

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_check() -> None:
    """The API should expose its initial readiness endpoint."""

    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["capabilities"]["upload"] is True
