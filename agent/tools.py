"""BrainTumor MRI Assistant允许调用的三个医学影像工具。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from rag.retriever import MedicalKnowledgeRetriever, RetrievalResponse
from report.generator import GeneratedReport, QwenReportGenerator
from report.template import MRIAnalysisInput


class AgentToolError(ValueError):
    """Agent工具输入或执行失败。"""


class MRIAnalyzerProtocol(Protocol):
    def invoke(self, feature_json: Mapping[str, Any]) -> dict[str, Any]:
        """解释MRI量化指标。"""


class ReportGeneratorProtocol(Protocol):
    def invoke(
        self,
        feature_json: Mapping[str, Any],
        case_id: str | None = None,
    ) -> GeneratedReport:
        """生成影像辅助报告。"""


class MedicalRAGProtocol(Protocol):
    def invoke(self, query: str, top_k: int = 5) -> RetrievalResponse:
        """检索医学知识。"""


class MRIAnalyzerTool:
    """将feature JSON转换为不含诊断推断的影像指标解释。"""

    name = "mri_analyzer"

    def invoke(self, feature_json: Mapping[str, Any]) -> dict[str, Any]:
        """校验并解释位置、体积、增强比例和水肿指标。"""

        analysis = MRIAnalysisInput.from_mapping(feature_json)
        explanations = [
            (
                f"主要位置：{analysis.location}。该位置来自AI分割区域的空间定位，"
                "应结合原始MRI由医师复核。"
            ),
            (
                f"Whole Tumor体积：{analysis.tumor_volume:.3f} cm³，"
                "表示分割mask中全部肿瘤相关标签的物理体积。"
            ),
            (
                f"增强区域占比：{analysis.enhancing_ratio:.2%}，"
                "表示Enhancing Tumor占Whole Tumor的体素比例；"
                "该比例本身不能确定病理类型或肿瘤分级。"
            ),
            (
                "水肿区域：AI分割检出。"
                if analysis.edema
                else "水肿区域：当前AI分割未检出。"
            ),
        ]
        optional_metrics = {
            key: value
            for key, value in analysis.to_prompt_payload().items()
            if key
            not in {
                "location",
                "tumor_volume",
                "enhancing_ratio",
                "edema",
            }
        }
        return {
            "location": analysis.location,
            "tumor_volume_cm3": analysis.tumor_volume,
            "enhancing_ratio": analysis.enhancing_ratio,
            "edema": analysis.edema,
            "optional_metrics": optional_metrics,
            "metric_explanations": explanations,
            "interpretation_boundary": (
                "以上为AI分割量化指标解释，不构成疾病诊断、病理分型或分级。"
            ),
        }


class ReportGeneratorTool:
    """Qwen安全报告生成器的Agent工具适配层。"""

    name = "report_generator"

    def __init__(self, generator: QwenReportGenerator) -> None:
        self.generator = generator

    def invoke(
        self,
        feature_json: Mapping[str, Any],
        case_id: str | None = None,
    ) -> GeneratedReport:
        """生成五章节MRI影像辅助报告。"""

        return self.generator.generate(feature_json, case_id=case_id)


class MedicalRAGTool:
    """带版本和页码引用的医学知识检索工具。"""

    name = "medical_rag"

    def __init__(self, retriever: MedicalKnowledgeRetriever) -> None:
        self.retriever = retriever

    def invoke(self, query: str, top_k: int = 5) -> RetrievalResponse:
        """检索与医生问题相关的医学原文片段。"""

        return self.retriever.retrieve(query=query, top_k=top_k)


@dataclass(frozen=True, slots=True)
class AgentTools:
    """LangGraph节点可调用的白名单工具集合。"""

    mri_analyzer: MRIAnalyzerProtocol
    report_generator: ReportGeneratorProtocol
    medical_rag: MedicalRAGProtocol
