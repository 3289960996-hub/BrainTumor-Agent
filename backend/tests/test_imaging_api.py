"""MRI Assistant后端端点和分析编排测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent.assistant import AssistantResponse
from backend.app.main import app
from backend.app.services.analysis import AnalysisResult, MRIAnalysisPipeline
from backend.app.services.dependencies import (
    get_analysis_pipeline,
    get_case_repository,
    get_chat_service,
    get_report_service,
    get_upload_service,
)
from backend.app.services.errors import CaseNotFoundError
from backend.app.services.storage import CasePaths, CaseRepository
from backend.app.services.upload import MRIUploadService
from report.generator import GeneratedReport
from report.template import MRIAnalysisInput

METRICS: dict[str, Any] = {
    "tumor_volume": 35.5,
    "tumor_core_volume": 16.2,
    "enhancing_volume": 8.1,
    "max_diameter": 42.0,
    "edema": True,
    "location": "left frontal",
    "edema_volume": 19.3,
    "tumor_core_ratio": 0.4563,
    "enhancing_ratio": 0.2282,
    "edema_ratio": 0.5437,
}


@pytest.fixture(autouse=True)
def _clear_dependency_overrides() -> None:
    yield
    app.dependency_overrides.clear()


def _nifti_file(path: Path, data: np.ndarray | None = None) -> Path:
    array = (
        np.zeros((5, 6, 7), dtype=np.float32)
        if data is None
        else np.asarray(data)
    )
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def _multipart_modalities(tmp_path: Path) -> dict[str, tuple[str, bytes, str]]:
    files: dict[str, tuple[str, bytes, str]] = {}
    for modality in ("t1", "t1ce", "t2", "flair"):
        path = _nifti_file(tmp_path / f"source_{modality}.nii.gz")
        files[modality] = (
            path.name,
            path.read_bytes(),
            "application/gzip",
        )
    return files


def test_upload_saves_four_standardized_modalities(tmp_path: Path) -> None:
    repository = CaseRepository(tmp_path / "data")
    service = MRIUploadService(repository, max_file_bytes=10 * 1024 * 1024)
    app.dependency_overrides[get_upload_service] = lambda: service

    response = TestClient(app).post(
        "/api/v1/upload",
        data={"case_id": "case-upload"},
        files=_multipart_modalities(tmp_path),
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["case_id"] == "case-upload"
    assert payload["status"] == "uploaded"
    assert set(payload["modalities"]) == {"t1", "t1ce", "t2", "flair"}
    paths = repository.require_case("case-upload")
    assert len(list(paths.raw.glob("*.nii.gz"))) == 4
    assert repository.read_status("case-upload")["status"] == "uploaded"


def test_upload_rejects_non_nifti_and_removes_partial_case(
    tmp_path: Path,
) -> None:
    repository = CaseRepository(tmp_path / "data")
    service = MRIUploadService(repository, max_file_bytes=1024)
    app.dependency_overrides[get_upload_service] = lambda: service
    files = {
        modality: (f"{modality}.txt", b"not nifti", "text/plain")
        for modality in ("t1", "t1ce", "t2", "flair")
    }

    response = TestClient(app).post(
        "/api/v1/upload",
        data={"case_id": "case-invalid"},
        files=files,
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_upload"
    assert not repository.paths("case-invalid").root.exists()


class FakeAnalysisPipeline:
    def __init__(self, mask_path: Path) -> None:
        self.mask_path = mask_path
        self.calls: list[str] = []

    def analyze(self, case_id: str) -> AnalysisResult:
        self.calls.append(case_id)
        return AnalysisResult(case_id, self.mask_path, METRICS)


def test_analyze_returns_mask_and_tumor_metrics(tmp_path: Path) -> None:
    mask = _nifti_file(tmp_path / "case-api.nii.gz")
    pipeline = FakeAnalysisPipeline(mask)
    app.dependency_overrides[get_analysis_pipeline] = lambda: pipeline

    response = TestClient(app).post(
        "/api/v1/analyze",
        json={"case_id": "case-api"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mask"]["filename"] == "case-api.nii.gz"
    assert payload["mask"]["label_space"] == "brats"
    assert payload["mask"]["download_url"].endswith(
        "/api/v1/cases/case-api/mask"
    )
    assert payload["tumor_metrics"]["tumor_volume"] == 35.5
    assert pipeline.calls == ["case-api"]


class FakePreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def prepare(self, paths: CasePaths) -> Path:
        self.calls += 1
        paths.processed_case.mkdir(parents=True, exist_ok=True)
        return paths.processed_case


class FakePredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, input_dir: Path, output_root: Path, case_id: str) -> Path:
        self.calls += 1
        labels = np.zeros((5, 5, 5), dtype=np.uint8)
        labels[1:4, 1:4, 1:4] = 2
        labels[2:4, 2:4, 2:4] = 1
        labels[3, 3, 3] = 4
        target = output_root / "brats_predictions" / f"{case_id}.nii.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        return _nifti_file(target, labels)


def test_analysis_pipeline_runs_preprocess_predict_and_measure(
    tmp_path: Path,
) -> None:
    repository = CaseRepository(tmp_path / "data")
    repository.create_case("case-pipeline")
    repository.write_status("case-pipeline", "uploaded")
    preprocessor = FakePreprocessor()
    predictor = FakePredictor()
    pipeline = MRIAnalysisPipeline(repository, preprocessor, predictor)

    result = pipeline.analyze("case-pipeline")
    cached = pipeline.analyze("case-pipeline")

    assert result.mask_path.is_file()
    assert result.metrics["tumor_volume"] == 0.027
    assert result.metrics["enhancing_volume"] == 0.001
    assert result.metrics["edema"] is True
    assert repository.read_status("case-pipeline")["status"] == "analyzed"
    assert cached.metrics == result.metrics
    assert preprocessor.calls == 1
    assert predictor.calls == 1


class FakeReportService:
    def generate(self, case_id: str) -> GeneratedReport:
        return GeneratedReport(
            content="影像表现提示：AI勾画区域需要医师复核。建议结合临床。",
            model="fake-qwen",
            request_id="request-1",
            analysis=MRIAnalysisInput.from_mapping(METRICS),
        )


def test_report_returns_assisted_report() -> None:
    app.dependency_overrides[get_report_service] = lambda: FakeReportService()

    response = TestClient(app).post(
        "/api/v1/report",
        json={"case_id": "case-report"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "report_ready"
    assert "建议结合临床" in payload["report"]
    assert payload["requires_human_review"] is True


class FakeChatService:
    def chat(
        self,
        question: str,
        case_id: str | None = None,
    ) -> AssistantResponse:
        return AssistantResponse(
            run_id="run-1",
            intent="mri_summary",
            answer="AI定量分析显示存在需要复核的区域。",
            tool_name="mri_analyzer",
            citations=(),
            safety_warnings=(),
            requires_human_review=True,
        )


def test_chat_returns_agent_answer() -> None:
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    response = TestClient(app).post(
        "/api/v1/chat",
        json={
            "case_id": "case-chat",
            "question": "总结该MRI分析结果",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "mri_analyzer"
    assert payload["intent"] == "mri_summary"
    assert payload["requires_human_review"] is True


def test_mask_download_uses_repository_artifact(tmp_path: Path) -> None:
    repository = CaseRepository(tmp_path / "data")
    paths = repository.create_case("case-mask")
    repository.write_status("case-mask", "analyzed")
    paths.mask.parent.mkdir(parents=True, exist_ok=True)
    _nifti_file(paths.mask)
    app.dependency_overrides[get_case_repository] = lambda: repository

    response = TestClient(app).get("/api/v1/cases/case-mask/mask")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/gzip"
    assert response.content == paths.mask.read_bytes()


def test_service_error_is_returned_as_stable_json() -> None:
    class MissingPipeline:
        def analyze(self, case_id: str) -> AnalysisResult:
            raise CaseNotFoundError(case_id)

    app.dependency_overrides[get_analysis_pipeline] = lambda: MissingPipeline()

    response = TestClient(app).post(
        "/api/v1/analyze",
        json={"case_id": "missing-case"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "case_not_found"
