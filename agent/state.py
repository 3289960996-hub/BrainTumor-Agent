"""BrainTumor MRI Assistant的可序列化LangGraph状态。"""

from typing import Any, Literal, TypedDict

AgentIntent = Literal[
    "mri_summary",
    "report_generation",
    "medical_knowledge",
    "missing_features",
    "unsupported",
]


class AgentState(TypedDict, total=False):
    """一次医生问答在LangGraph各节点之间传递的状态。"""

    run_id: str
    case_id: str
    user_query: str
    imaging_features: dict[str, Any]
    intent: AgentIntent
    tool_name: str
    tool_output: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]
    report_content: str
    final_response: str
    citations: list[str]
    tool_error: str
    safety_warnings: list[str]
    requires_human_review: bool
