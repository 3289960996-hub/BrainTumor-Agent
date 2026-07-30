"""Agent路由、工具调用与医学安全边界测试。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent.assistant import BrainTumorMRIAssistant
from agent.state import AgentIntent
from agent.tools import AgentTools, MRIAnalyzerTool
from rag.retriever import RetrievalResponse
from rag.schemas import RetrievedChunk
from report.generator import GeneratedReport
from report.template import MRIAnalysisInput

FEATURES: dict[str, Any] = {
    "location": "left frontal",
    "tumor_volume": 35.5,
    "tumor_core_volume": 16.2,
    "enhancing_volume": 8.1,
    "max_diameter": 42.0,
    "enhancing_ratio": 0.228,
    "edema": True,
}


class FakeModel:
    """记录调用次数的无网络模型。"""

    def __init__(
        self,
        *,
        classified_intent: AgentIntent = "unsupported",
        summary: str = "AI定量分析显示肿瘤相关区域体积为35.5 cm³，建议结合临床。",
    ) -> None:
        self.classified_intent = classified_intent
        self.summary = summary
        self.classify_calls = 0
        self.summary_calls = 0
        self.rag_calls = 0
        self.repair_calls = 0

    def classify_intent(
        self,
        user_query: str,
        has_features: bool,
    ) -> AgentIntent:
        self.classify_calls += 1
        return self.classified_intent

    def summarize_imaging(
        self,
        user_query: str,
        tool_output: Mapping[str, Any],
    ) -> str:
        self.summary_calls += 1
        return self.summary

    def answer_with_evidence(
        self,
        user_query: str,
        evidence_context: str,
    ) -> str:
        self.rag_calls += 1
        return (
            "增强区域可用于描述造影剂相关信号变化，并可作为随访时需要"
            "对照观察的影像指标之一。[资料1] 建议结合临床。"
        )

    def repair_safe_response(
        self,
        unsafe_response: str,
        violations: Sequence[str],
    ) -> str:
        self.repair_calls += 1
        return "影像表现提示存在需关注的定量变化，建议结合临床并由医师审核。"


class FailingClassifierModel(FakeModel):
    def classify_intent(
        self,
        user_query: str,
        has_features: bool,
    ) -> AgentIntent:
        self.classify_calls += 1
        raise RuntimeError("model unavailable")


class CountingMRIAnalyzer:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = MRIAnalyzerTool()

    def invoke(self, feature_json: Mapping[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return self.delegate.invoke(feature_json)


class FakeReportGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(
        self,
        feature_json: Mapping[str, Any],
        case_id: str | None = None,
    ) -> GeneratedReport:
        self.calls += 1
        analysis = MRIAnalysisInput.from_mapping(feature_json)
        return GeneratedReport(
            content=(
                "# MRI影像辅助报告\n\n"
                "AI分割结果仅供辅助参考。\n\n"
                "影像表现提示：存在AI勾画区域。\n\n"
                "建议结合临床，并由影像科医师复核。"
            ),
            model="fake-qwen",
            request_id="test-request",
            analysis=analysis,
        )


class FakeMedicalRAG:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, query: str, top_k: int = 5) -> RetrievalResponse:
        self.calls += 1
        return RetrievalResponse(
            query=query,
            chunks=(
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    text="增强区域是肿瘤MRI评估与随访中需要结合序列对照观察的指标。",
                    score=0.91,
                    citation="MRI胶质瘤影像表现（2025版），第12页",
                    metadata={"topic": "mri_glioma"},
                ),
            ),
        )


def _assistant(
    model: FakeModel | None = None,
) -> tuple[
    BrainTumorMRIAssistant,
    FakeModel,
    CountingMRIAnalyzer,
    FakeReportGenerator,
    FakeMedicalRAG,
]:
    resolved_model = model or FakeModel()
    mri = CountingMRIAnalyzer()
    report = FakeReportGenerator()
    rag = FakeMedicalRAG()
    assistant = BrainTumorMRIAssistant(
        tools=AgentTools(
            mri_analyzer=mri,
            report_generator=report,
            medical_rag=rag,
        ),
        model=resolved_model,
    )
    return assistant, resolved_model, mri, report, rag


def test_mri_analyzer_explains_features_without_diagnosis() -> None:
    output = MRIAnalyzerTool().invoke(FEATURES)

    assert output["tumor_volume_cm3"] == 35.5
    assert output["enhancing_ratio"] == 0.228
    assert output["edema"] is True
    assert "不能确定病理类型或肿瘤分级" in " ".join(
        output["metric_explanations"]
    )


def test_summary_question_routes_only_to_mri_tool() -> None:
    assistant, model, mri, report, rag = _assistant()

    response = assistant.ask("总结该MRI分析结果", FEATURES, case_id="case-001")

    assert response.intent == "mri_summary"
    assert response.tool_name == "mri_analyzer"
    assert mri.calls == 1
    assert report.calls == 0
    assert rag.calls == 0
    assert model.summary_calls == 1
    assert "不提供疾病确诊" in response.answer
    assert response.requires_human_review is True


def test_enhancing_question_routes_to_rag_with_citation() -> None:
    assistant, model, mri, report, rag = _assistant()

    response = assistant.ask("为什么需要关注增强区域？")

    assert response.intent == "medical_knowledge"
    assert response.tool_name == "medical_rag"
    assert rag.calls == 1
    assert mri.calls == 0
    assert report.calls == 0
    assert model.rag_calls == 1
    assert response.citations == ("MRI胶质瘤影像表现（2025版），第12页",)
    assert "[资料1]" in response.answer


def test_report_question_routes_only_to_report_tool() -> None:
    assistant, _, mri, report, rag = _assistant()

    response = assistant.ask("请生成辅助报告", FEATURES)

    assert response.intent == "report_generation"
    assert response.tool_name == "report_generator"
    assert report.calls == 1
    assert mri.calls == 0
    assert rag.calls == 0
    assert "# MRI影像辅助报告" in response.answer


def test_summary_without_features_returns_controlled_message() -> None:
    assistant, _, mri, report, rag = _assistant()

    response = assistant.ask("总结该MRI分析结果")

    assert response.intent == "missing_features"
    assert response.tool_name is None
    assert "feature JSON" in response.answer
    assert mri.calls == 0
    assert report.calls == 0
    assert rag.calls == 0


def test_unsafe_model_summary_is_rewritten() -> None:
    unsafe_model = FakeModel(summary="明确诊断为高级别胶质瘤。")
    assistant, model, _, _, _ = _assistant(unsafe_model)

    response = assistant.ask("总结该MRI分析结果", FEATURES)

    assert model.repair_calls == 1
    assert "明确诊断为" not in response.answer
    assert "建议结合临床" in response.answer
    assert response.safety_warnings


def test_ambiguous_question_uses_qwen_intent_classifier() -> None:
    model = FakeModel(classified_intent="medical_knowledge")
    assistant, model, _, _, rag = _assistant(model)

    response = assistant.ask("请解释这个概念")

    assert model.classify_calls == 1
    assert response.intent == "medical_knowledge"
    assert rag.calls == 1


def test_classifier_failure_returns_controlled_response() -> None:
    model = FailingClassifierModel()
    assistant, _, mri, report, rag = _assistant(model)

    response = assistant.ask("请解释这个概念")

    assert response.intent == "unsupported"
    assert "问题意图识别失败" in response.answer
    assert mri.calls == 0
    assert report.calls == 0
    assert rag.calls == 0
