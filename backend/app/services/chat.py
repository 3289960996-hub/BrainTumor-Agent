"""Qwen MRI Assistant问答应用服务。"""

from __future__ import annotations

from pathlib import Path

from agent.assistant import (
    AssistantResponse,
    BrainTumorMRIAssistant,
    create_default_assistant,
)
from agent.qwen import (
    AgentModelError,
    QwenAgentClient,
    QwenAgentConfig,
)
from backend.app.services.errors import (
    ExternalServiceError,
    ServiceConfigurationError,
)
from backend.app.services.storage import CaseRepository


class MedicalAgentChatService:
    """向Agent注入当前病例指标；无病例时仅允许知识类问答。"""

    def __init__(
        self,
        repository: CaseRepository,
        rag_index_path: str | Path,
        assistant: BrainTumorMRIAssistant | None = None,
        config: QwenAgentConfig | None = None,
    ) -> None:
        self.repository = repository
        self.rag_index_path = Path(rag_index_path)
        self._assistant = assistant
        self.config = config

    def _get_assistant(self) -> BrainTumorMRIAssistant:
        if self._assistant is None:
            try:
                model = QwenAgentClient(config=self.config)
                self._assistant = create_default_assistant(
                    self.rag_index_path,
                    model=model,
                )
            except AgentModelError as exc:
                raise ServiceConfigurationError(str(exc)) from exc
        return self._assistant

    def chat(
        self,
        question: str,
        case_id: str | None = None,
    ) -> AssistantResponse:
        features = None
        if case_id is not None:
            self.repository.require_case(case_id)
            features = self.repository.load_features(case_id)
        try:
            return self._get_assistant().ask(
                user_query=question.strip(),
                feature_json=features,
                case_id=case_id,
            )
        except AgentModelError as exc:
            raise ExternalServiceError("Qwen-plus Agent调用失败") from exc
