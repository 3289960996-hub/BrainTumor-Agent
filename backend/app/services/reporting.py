"""病例辅助报告应用服务。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Protocol

from backend.app.services.errors import (
    CaseStateError,
    ExternalServiceError,
    ServiceConfigurationError,
)
from backend.app.services.storage import CaseRepository
from report.generator import (
    EditedReport,
    GeneratedReport,
    QwenReportGenerator,
    ReportConfig,
    ReportGenerationError,
)


class ReportGeneratorProtocol(Protocol):
    def generate(
        self,
        analysis_data: dict,
        case_id: str | None = None,
    ) -> GeneratedReport:
        """生成影像辅助报告。"""


class ReportEditorProtocol(Protocol):
    def edit(
        self,
        current_report: str,
        instruction: str,
        analysis: dict,
    ) -> EditedReport:
        """Generate a report edit proposal."""


class MedicalReportService:
    """从病例指标生成并持久化需要人工审核的辅助报告。"""

    def __init__(
        self,
        repository: CaseRepository,
        generator: ReportGeneratorProtocol | None = None,
        config: ReportConfig | None = None,
    ) -> None:
        self.repository = repository
        self._generator = generator
        self.config = config

    def _get_generator(self) -> ReportGeneratorProtocol:
        if self._generator is None:
            try:
                self._generator = QwenReportGenerator(config=self.config)
            except ReportGenerationError as exc:
                raise ServiceConfigurationError(str(exc)) from exc
        return self._generator

    def generate(self, case_id: str) -> GeneratedReport:
        with self.repository.case_lock(case_id):
            self.repository.require_case(case_id)
            features = self.repository.load_features(case_id)
            if features is None:
                raise CaseStateError("请先调用/analyze生成肿瘤量化指标")
            try:
                generated = self._get_generator().generate(
                    features,
                    case_id=case_id,
                )
            except ReportGenerationError as exc:
                raise ExternalServiceError("Qwen-plus辅助报告生成失败") from exc
            self.repository.save_report(case_id, generated.content)
            self.repository.write_status(
                case_id,
                "report_ready",
                extra={
                    "report_filename": "report.md",
                    "report_stale": False,
                },
            )
            return generated


class MedicalReportEditingService:
    """先生成修改建议，医生确认后才写入正式报告。"""

    def __init__(
        self,
        repository: CaseRepository,
        editor: ReportEditorProtocol | None = None,
        config: ReportConfig | None = None,
    ) -> None:
        self.repository = repository
        self._editor = editor
        self.config = config

    def _get_editor(self) -> ReportEditorProtocol:
        if self._editor is None:
            try:
                self._editor = QwenReportGenerator(config=self.config)
            except ReportGenerationError as exc:
                raise ServiceConfigurationError(str(exc)) from exc
        return self._editor

    def propose(self, case_id: str, instruction: str) -> dict:
        with self.repository.case_lock(case_id):
            paths = self.repository.require_case(case_id)
            if not paths.report.is_file():
                raise CaseStateError("请先生成辅助报告，再进行报告协作编辑")
            current_report = paths.report.read_text(encoding="utf-8")
            features = self.repository.load_features(case_id)
            if features is None:
                raise CaseStateError("当前病例缺少定量分析结果")
            try:
                edited = self._get_editor().edit(
                    current_report=current_report,
                    instruction=instruction,
                    analysis=features,
                )
            except ReportGenerationError as exc:
                raise ExternalServiceError(f"报告修改建议生成失败：{exc}") from exc
            suggestion_id = (
                "suggest-"
                + hashlib.sha256(
                    (case_id + instruction + edited.content).encode("utf-8")
                ).hexdigest()[:16]
            )
            protected = {
                key: value
                for key, value in features.items()
                if key
                in {
                    "tumor_volume",
                    "tumor_core_volume",
                    "enhancing_volume",
                    "max_diameter",
                    "edema",
                    "location",
                    "edema_volume",
                }
            }
            payload = {
                "case_id": case_id,
                "suggestion_id": suggestion_id,
                "status": "pending",
                "instruction": instruction.strip(),
                "current_report": current_report,
                "proposed_report": edited.content,
                "change_summary": list(edited.change_summary),
                "protected_metrics": protected,
                "base_sha256": _sha256_text(current_report),
                "created_at": datetime.now(UTC).isoformat(),
            }
            self.repository.save_report_proposal(case_id, suggestion_id, payload)
            return payload

    def apply(self, case_id: str, suggestion_id: str) -> dict:
        with self.repository.case_lock(case_id):
            paths = self.repository.require_case(case_id)
            proposal = self.repository.load_report_proposal(case_id, suggestion_id)
            if proposal.get("status") != "pending":
                raise CaseStateError("该报告修改建议已经处理过")
            current_report = (
                paths.report.read_text(encoding="utf-8") if paths.report.is_file() else ""
            )
            if _sha256_text(current_report) != proposal.get("base_sha256"):
                raise CaseStateError("报告已发生变化，请重新生成修改建议")
            report = str(proposal.get("proposed_report", "")).strip()
            if not report:
                raise CaseStateError("报告修改建议内容为空")
            revision_id = self.repository.save_report_revision(case_id, current_report)
            self.repository.save_report(case_id, report)
            proposal["status"] = "applied"
            proposal["applied_at"] = datetime.now(UTC).isoformat()
            self.repository.save_report_proposal(case_id, suggestion_id, proposal)
            self.repository.write_status(
                case_id,
                "report_ready",
                extra={
                    "report_filename": "report.md",
                    "report_stale": False,
                    "last_report_revision": revision_id,
                },
            )
            return {"case_id": case_id, "revision_id": revision_id, "report": report}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
