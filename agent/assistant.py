"""BrainTumor MRI Assistant的应用入口与命令行。"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.qwen import AgentLanguageModel, AgentModelError, QwenAgentClient
from agent.state import AgentIntent, AgentState
from agent.tools import (
    AgentTools,
    MedicalRAGTool,
    MRIAnalyzerTool,
    ReportGeneratorTool,
)
from agent.workflow import build_workflow
from rag.retriever import MedicalKnowledgeRetriever
from report.generator import QwenReportGenerator, ReportConfig


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """Web API和前端可直接消费的Agent响应。"""

    run_id: str
    intent: AgentIntent
    answer: str
    tool_name: str | None
    citations: tuple[str, ...]
    safety_warnings: tuple[str, ...]
    requires_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为标准JSON对象。"""

        return asdict(self)


class BrainTumorMRIAssistant:
    """面向医生的医学影像辅助Agent。"""

    def __init__(self, tools: AgentTools, model: AgentLanguageModel) -> None:
        self.tools = tools
        self.model = model
        self.graph = build_workflow(tools=tools, model=model)

    def ask(
        self,
        user_query: str,
        feature_json: dict[str, Any] | None = None,
        case_id: str | None = None,
        run_id: str | None = None,
    ) -> AssistantResponse:
        """执行一次受控Agent工作流。"""

        resolved_run_id = run_id or str(uuid.uuid4())
        initial_state: AgentState = {
            "run_id": resolved_run_id,
            "user_query": user_query,
            "imaging_features": feature_json or {},
            "requires_human_review": True,
        }
        if case_id:
            initial_state["case_id"] = case_id
        final_state = self.graph.invoke(initial_state)
        return AssistantResponse(
            run_id=resolved_run_id,
            intent=final_state.get("intent", "unsupported"),
            answer=final_state.get("final_response", ""),
            tool_name=final_state.get("tool_name"),
            citations=tuple(final_state.get("citations", [])),
            safety_warnings=tuple(final_state.get("safety_warnings", [])),
            requires_human_review=True,
        )


def create_default_assistant(
    rag_index_path: str | Path | None = None,
    model: QwenAgentClient | None = None,
) -> BrainTumorMRIAssistant:
    """根据环境变量创建共享Qwen客户端和三个默认工具。"""

    resolved_model = model or QwenAgentClient()
    report_config = ReportConfig(
        api_key=resolved_model.config.api_key,
        base_url=resolved_model.config.base_url,
        model=resolved_model.config.model,
        temperature=float(os.getenv("BTA_REPORT_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("BTA_REPORT_MAX_TOKENS", "800")),
        timeout_seconds=resolved_model.config.timeout_seconds,
        enable_data_inspection=resolved_model.config.enable_data_inspection,
    )
    report_generator = QwenReportGenerator(
        config=report_config,
        client=resolved_model.client,
    )
    index_path = rag_index_path or os.getenv(
        "BTA_FAISS_INDEX_PATH",
        "./runtime/faiss",
    )
    tools = AgentTools(
        mri_analyzer=MRIAnalyzerTool(),
        report_generator=ReportGeneratorTool(report_generator),
        medical_rag=MedicalRAGTool(MedicalKnowledgeRetriever(index_path)),
    )
    return BrainTumorMRIAssistant(tools=tools, model=resolved_model)


def _load_feature_json(path: str | Path | None) -> dict[str, Any] | None:
    """读取可选feature JSON。"""

    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"feature JSON不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("feature JSON根节点必须是object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """创建Assistant命令行参数。"""

    parser = argparse.ArgumentParser(description="BrainTumor MRI Assistant")
    parser.add_argument("--question", required=True, help="医生问题")
    parser.add_argument("--feature-json", default=None, help="当前病例MRI分析JSON")
    parser.add_argument("--case-id", default=None, help="可选去标识化病例编号")
    parser.add_argument(
        "--rag-index",
        default=os.getenv("BTA_FAISS_INDEX_PATH", "./runtime/faiss"),
        help="医学知识FAISS索引目录",
    )
    parser.add_argument("--json", action="store_true", help="输出标准JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行一次医生问答。"""

    args = build_parser().parse_args(argv)
    try:
        features = _load_feature_json(args.feature_json)
        assistant = create_default_assistant(args.rag_index)
        response = assistant.ask(
            user_query=args.question,
            feature_json=features,
            case_id=args.case_id,
        )
    except (OSError, ValueError, json.JSONDecodeError, AgentModelError) as exc:
        print(f"Agent执行失败：{exc}")
        return 1

    if args.json:
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(response.answer)
        if response.citations:
            print("\n资料来源：")
            for citation in response.citations:
                print(f"- {citation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
