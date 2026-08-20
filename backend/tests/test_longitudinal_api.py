"""Longitudinal quantitative comparison API tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient

from backend.app.api.routes import comparisons as comparison_routes
from backend.app.main import app
from backend.app.services.dependencies import (
    get_case_repository,
    get_comparison_task_repository,
    get_longitudinal_comparison_service,
)
from backend.app.services.storage import CaseRepository
from backend.app.tasks import comparison as comparison_task_module
from longitudinal.service import LongitudinalComparisonService, comparison_id_for
from longitudinal.storage import ComparisonRepository, ComparisonTaskRepository

BASELINE: dict[str, Any] = {
    "tumor_volume": 40.0,
    "tumor_core_volume": 20.0,
    "enhancing_volume": 8.0,
    "max_diameter": 50.0,
    "edema": True,
    "location": "left frontal",
    "edema_volume": 20.0,
    "tumor_core_ratio": 0.5,
    "enhancing_ratio": 0.2,
    "edema_ratio": 0.5,
}

FOLLOWUP: dict[str, Any] = {
    "tumor_volume": 50.0,
    "tumor_core_volume": 18.0,
    "enhancing_volume": 10.0,
    "max_diameter": 55.0,
    "edema": True,
    "location": "left frontal",
    "edema_volume": 32.0,
    "tumor_core_ratio": 0.36,
    "enhancing_ratio": 0.2,
    "edema_ratio": 0.64,
}


@pytest.fixture(autouse=True)
def _clear_dependency_overrides() -> None:
    yield
    app.dependency_overrides.clear()


def _services(tmp_path: Path) -> tuple[CaseRepository, LongitudinalComparisonService]:
    cases = CaseRepository(tmp_path / "data")
    comparisons = ComparisonRepository(tmp_path / "data")
    return cases, LongitudinalComparisonService(cases, comparisons)


def _analyzed_case(
    repository: CaseRepository,
    case_id: str,
    metrics: dict[str, Any],
) -> None:
    repository.create_case(case_id)
    repository.save_features(case_id, metrics)
    repository.write_status(case_id, "analyzed")


def _spatial_case(
    repository: CaseRepository,
    case_id: str,
    metrics: dict[str, Any],
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    paths = repository.create_case(case_id)
    image_path = paths.raw / f"{case_id}_t1ce.nii.gz"
    mask_path = paths.mask
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(image.astype(np.float32)), str(image_path))
    sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.uint8)), str(mask_path))
    repository.save_features(case_id, metrics)
    repository.write_status(
        case_id,
        "analyzed",
        extra={"modalities": {"t1ce": image_path.name}},
    )


def test_lists_only_analyzed_cases_for_comparison(tmp_path: Path) -> None:
    repository, _ = _services(tmp_path)
    _analyzed_case(repository, "case-baseline", BASELINE)
    repository.create_case("case-uploaded")
    repository.write_status("case-uploaded", "uploaded")
    app.dependency_overrides[get_case_repository] = lambda: repository

    response = TestClient(app).get("/api/v1/cases?analyzed_only=true")

    assert response.status_code == 200
    assert [item["case_id"] for item in response.json()["cases"]] == [
        "case-baseline"
    ]
    assert response.json()["cases"][0]["has_metrics"] is True


def test_creates_and_restores_deterministic_longitudinal_comparison(
    tmp_path: Path,
) -> None:
    repository, service = _services(tmp_path)
    _analyzed_case(repository, "case-baseline", BASELINE)
    _analyzed_case(repository, "case-followup", FOLLOWUP)
    app.dependency_overrides[get_longitudinal_comparison_service] = lambda: service

    response = TestClient(app).post(
        "/api/v1/comparisons",
        json={
            "patient_group_id": "subject-001",
            "baseline_case_id": "case-baseline",
            "followup_case_id": "case-followup",
            "baseline_study_date": "2026-01-01",
            "followup_study_date": "2026-04-01",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["comparison_id"].startswith("comparison-")
    assert payload["interval_days"] == 90
    assert payload["spatial_comparison_available"] is False
    assert payload["location_consistent"] is True
    changes = {item["key"]: item for item in payload["metrics"]}
    assert changes["tumor_volume"]["absolute_change"] == 10.0
    assert changes["tumor_volume"]["percent_change"] == 25.0
    assert changes["tumor_core_volume"]["direction"] == "decreased"
    assert changes["edema_ratio"]["percentage_point_change"] == 14.0
    assert changes["edema_ratio"]["percent_change"] is None

    restored = TestClient(app).get(
        f"/api/v1/comparisons/{payload['comparison_id']}"
    )
    assert restored.status_code == 200
    assert restored.json() == payload
    comparison_file = (
        tmp_path
        / "data"
        / "comparisons"
        / payload["comparison_id"]
        / "comparison.json"
    )
    assert comparison_file.is_file()


@pytest.mark.parametrize(
    ("request_update", "expected_message"),
    [
        (
            {"followup_case_id": "case-baseline"},
            "基线病例和随访病例不能是同一个病例",
        ),
        (
            {"followup_study_date": "2025-12-31"},
            "随访检查日期必须晚于基线检查日期",
        ),
    ],
)
def test_rejects_invalid_case_pair_or_date_order(
    tmp_path: Path,
    request_update: dict[str, str],
    expected_message: str,
) -> None:
    repository, service = _services(tmp_path)
    _analyzed_case(repository, "case-baseline", BASELINE)
    _analyzed_case(repository, "case-followup", FOLLOWUP)
    app.dependency_overrides[get_longitudinal_comparison_service] = lambda: service
    request = {
        "patient_group_id": "subject-001",
        "baseline_case_id": "case-baseline",
        "followup_case_id": "case-followup",
        "baseline_study_date": "2026-01-01",
        "followup_study_date": "2026-04-01",
    }
    request.update(request_update)

    response = TestClient(app).post("/api/v1/comparisons", json=request)

    assert response.status_code == 409
    assert response.json()["detail"]["message"] == expected_message


def test_rejects_unanalyzed_case_and_handles_zero_baseline(tmp_path: Path) -> None:
    repository, service = _services(tmp_path)
    zero_baseline = dict(BASELINE, enhancing_volume=0.0, enhancing_ratio=0.0)
    _analyzed_case(repository, "case-baseline", zero_baseline)
    repository.create_case("case-followup")
    repository.write_status("case-followup", "uploaded")
    app.dependency_overrides[get_longitudinal_comparison_service] = lambda: service
    request = {
        "patient_group_id": "subject-001",
        "baseline_case_id": "case-baseline",
        "followup_case_id": "case-followup",
        "baseline_study_date": "2026-01-01",
        "followup_study_date": "2026-04-01",
    }

    missing = TestClient(app).post("/api/v1/comparisons", json=request)
    assert missing.status_code == 409
    assert "尚未完成定量分析" in missing.json()["detail"]["message"]

    repository.save_features("case-followup", FOLLOWUP)
    repository.write_status("case-followup", "analyzed")
    response = TestClient(app).post("/api/v1/comparisons", json=request)
    changes = {item["key"]: item for item in response.json()["metrics"]}
    assert changes["enhancing_volume"]["percent_change"] is None
    assert changes["enhancing_ratio"]["percentage_point_change"] == 20.0


def test_creates_quality_gated_spatial_change_artifacts(tmp_path: Path) -> None:
    repository, service = _services(tmp_path)
    coordinates = np.indices((32, 32, 32), dtype=np.float32)
    distance = sum((axis - 15.5) ** 2 for axis in coordinates)
    image = np.exp(-distance / 90.0) * 100
    baseline_mask = np.zeros(image.shape, dtype=np.uint8)
    followup_mask = np.zeros(image.shape, dtype=np.uint8)
    baseline_mask[11:19, 11:19, 11:19] = 4
    followup_mask[13:22, 11:19, 11:19] = 4
    _spatial_case(repository, "case-baseline", BASELINE, image, baseline_mask)
    _spatial_case(repository, "case-followup", FOLLOWUP, image, followup_mask)
    app.dependency_overrides[get_longitudinal_comparison_service] = lambda: service

    response = TestClient(app).post(
        "/api/v1/comparisons",
        json={
            "patient_group_id": "subject-spatial",
            "baseline_case_id": "case-baseline",
            "followup_case_id": "case-followup",
            "baseline_study_date": "2026-01-01",
            "followup_study_date": "2026-04-01",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["spatial_comparison_available"] is True
    spatial = payload["spatial_comparison"]
    assert spatial["status"] == "quality_passed"
    assert spatial["quality"]["passed"] is True
    assert set(spatial["changes"]) == {"wt", "tc", "et"}
    assert spatial["changes"]["et"]["new_voxels"] > 0
    assert spatial["changes"]["et"]["resolved_voxels"] > 0
    for key in ("registered_followup_t1ce", "wt_change", "tc_change", "et_change"):
        artifact = TestClient(app).get(spatial["artifacts"][key])
        assert artifact.status_code == 200


def test_submits_polls_and_cancels_comparison_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, service = _services(tmp_path)
    tasks = ComparisonTaskRepository(tmp_path / "data")
    _analyzed_case(repository, "case-baseline", BASELINE)
    _analyzed_case(repository, "case-followup", FOLLOWUP)
    app.dependency_overrides[get_longitudinal_comparison_service] = lambda: service
    app.dependency_overrides[get_comparison_task_repository] = lambda: tasks
    sent: list[tuple[str, list[str], str]] = []
    monkeypatch.setattr(comparison_routes, "_worker_available", lambda: True)
    monkeypatch.setattr(
        comparison_routes.celery_app,
        "send_task",
        lambda name, args, task_id: sent.append((name, args, task_id)),
    )
    request = {
        "patient_group_id": "subject-async",
        "baseline_case_id": "case-baseline",
        "followup_case_id": "case-followup",
        "baseline_study_date": "2026-01-01",
        "followup_study_date": "2026-04-01",
    }

    submitted = TestClient(app).post("/api/v1/comparison-tasks", json=request)

    assert submitted.status_code == 202
    task = submitted.json()
    assert task["status"] == "queued"
    assert task["progress"] == 0
    assert sent == [("comparison.run", [task["task_id"]], task["task_id"])]
    polled = TestClient(app).get(f"/api/v1/comparison-tasks/{task['task_id']}")
    assert polled.status_code == 200
    cancelled = TestClient(app).post(
        f"/api/v1/comparison-tasks/{task['task_id']}/cancel"
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


def test_comparison_worker_persists_progress_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, service = _services(tmp_path)
    tasks = ComparisonTaskRepository(tmp_path / "data")
    _analyzed_case(repository, "case-baseline", BASELINE)
    _analyzed_case(repository, "case-followup", FOLLOWUP)
    baseline_date = date(2026, 1, 1)
    followup_date = date(2026, 4, 1)
    comparison_id = comparison_id_for(
        "subject-worker",
        "case-baseline",
        "case-followup",
        baseline_date,
        followup_date,
    )
    task, _ = tasks.create(
        comparison_id=comparison_id,
        request_payload={
            "patient_group_id": "subject-worker",
            "baseline_case_id": "case-baseline",
            "followup_case_id": "case-followup",
            "baseline_study_date": baseline_date.isoformat(),
            "followup_study_date": followup_date.isoformat(),
        },
    )
    monkeypatch.setattr(
        comparison_task_module,
        "get_comparison_task_repository",
        lambda: tasks,
    )
    monkeypatch.setattr(
        comparison_task_module,
        "get_longitudinal_comparison_service",
        lambda: service,
    )

    result = comparison_task_module.run_comparison.run(task["task_id"])

    assert result["status"] == "succeeded"
    assert result["progress"] == 100
    assert result["comparison_id"] == comparison_id
    assert service.get(comparison_id)["comparison_version"] == 2
