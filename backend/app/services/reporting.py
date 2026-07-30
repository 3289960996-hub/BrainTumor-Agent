"""病例辅助报告应用服务。"""

from __future__ import annotations

from typing import Protocol

from backend.app.services.errors import (
    CaseStateError,
    ExternalServiceError,
    ServiceConfigurationError,
)
from backend.app.services.storage import CaseRepository
from report.generator import (
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
                extra={"report_filename": "report.md"},
            )
            return generated
