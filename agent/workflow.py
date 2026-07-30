"""BrainTumor MRI Assistant受控LangGraph工作流。"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Literal, cast

from langgraph.graph import END, START, StateGraph

from agent.qwen import AgentLanguageModel
from agent.state import AgentIntent, AgentState
from agent.tools import AgentTools

SUMMARY_KEYWORDS = ("总结", "分析结果", "影像指标", "量化结果", "体积", "位置")
REPORT_KEYWORDS = ("生成报告", "辅助报告", "影像报告", "出报告")
KNOWLEDGE_KEYWORDS = (
    "为什么",
    "意义",
    "指南",
    "标准",
    "依据",
    "WHO",
    "NCCN",
    "随访",
    "如何评价",
    "需要关注",
)

UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:明确|最终|临床)?诊断为"),
    re.compile(r"(?:可以|可|已经|已)确诊为"),
    re.compile(r"考虑为(?:胶质|脑膜|转移|淋巴|生殖细胞|室管膜)"),
    re.compile(r"WHO\s*[ⅠIⅡVⅢIV1-4]+级", re.IGNORECASE),
)

SAFETY_NOTICE = (
    "> 本助手定位为医学影像辅助助手，不提供疾病确诊；"
    "结果须由专业医师结合原始影像、临床资料及必要检查审核。"
)


def route_user_query(
    user_query: str,
    has_features: bool,
    model: AgentLanguageModel,
) -> AgentIntent:
    """使用高置信规则优先路由，模糊问题交给Qwen分类。"""

    query = user_query.strip()
    if any(keyword in query for keyword in REPORT_KEYWORDS):
        intent: AgentIntent = "report_generation"
    elif any(keyword in query for keyword in KNOWLEDGE_KEYWORDS):
        intent = "medical_knowledge"
    elif any(keyword in query for keyword in SUMMARY_KEYWORDS):
        intent = "mri_summary"
    else:
        intent = model.classify_intent(query, has_features=has_features)

    if intent in {"mri_summary", "report_generation"} and not has_features:
        return "missing_features"
    return intent


def find_safety_violations(text: str) -> tuple[str, ...]:
    """检测直接确诊、病理推断和分级式表达。"""

    return tuple(
        f"包含禁止的诊断式表达：{pattern.pattern}"
        for pattern in UNSAFE_PATTERNS
        if pattern.search(text)
    )


def build_workflow(*, tools: AgentTools, model: AgentLanguageModel):
    """构建MRI总结、报告生成和RAG问答三条受控路径。"""

    def classify_node(state: AgentState) -> AgentState:
        query = state.get("user_query", "").strip()
        if not query:
            return {
                "intent": "unsupported",
                "tool_error": "用户问题不能为空",
            }
        try:
            intent = route_user_query(
                user_query=query,
                has_features=bool(state.get("imaging_features")),
                model=model,
            )
        except Exception as exc:
            return {
                "intent": "unsupported",
                "tool_error": f"问题意图识别失败：{exc}",
            }
        return {"intent": intent}

    def route_after_classify(
        state: AgentState,
    ) -> Literal[
        "mri_tool",
        "report_tool",
        "rag_tool",
        "unsupported",
    ]:
        intent = state.get("intent", "unsupported")
        mapping = {
            "mri_summary": "mri_tool",
            "report_generation": "report_tool",
            "medical_knowledge": "rag_tool",
            "missing_features": "unsupported",
            "unsupported": "unsupported",
        }
        return cast(
            Literal["mri_tool", "report_tool", "rag_tool", "unsupported"],
            mapping[intent],
        )

    def mri_tool_node(state: AgentState) -> AgentState:
        try:
            output = tools.mri_analyzer.invoke(state["imaging_features"])
            return {
                "tool_name": "mri_analyzer",
                "tool_output": output,
            }
        except Exception as exc:
            return {
                "tool_name": "mri_analyzer",
                "tool_error": f"MRI Analyzer执行失败：{exc}",
            }

    def mri_synthesis_node(state: AgentState) -> AgentState:
        if state.get("tool_error"):
            return {
                "final_response": (
                    f"无法总结当前MRI分析结果：{state['tool_error']}。"
                    "请检查feature JSON后重试。"
                )
            }
        try:
            answer = model.summarize_imaging(
                state["user_query"],
                state["tool_output"],
            )
            return {"final_response": answer}
        except Exception as exc:
            return {
                "final_response": (
                    f"影像指标已完成解析，但摘要生成失败：{exc}。"
                    "请由影像科医师直接查看结构化指标。"
                ),
                "safety_warnings": ["Qwen摘要生成失败"],
            }

    def report_tool_node(state: AgentState) -> AgentState:
        try:
            generated = tools.report_generator.invoke(
                state["imaging_features"],
                case_id=state.get("case_id"),
            )
            return {
                "tool_name": "report_generator",
                "tool_output": generated.to_agent_payload(),
                "report_content": generated.content,
                "final_response": generated.content,
            }
        except Exception as exc:
            return {
                "tool_name": "report_generator",
                "tool_error": f"Report Generator执行失败：{exc}",
                "final_response": (
                    f"辅助报告生成失败：{exc}。请检查feature JSON和Qwen配置。"
                ),
            }

    def rag_tool_node(state: AgentState) -> AgentState:
        try:
            result = tools.medical_rag.invoke(state["user_query"], top_k=5)
            return {
                "tool_name": "medical_rag",
                "tool_output": result.to_dict(),
                "retrieved_evidence": [
                    asdict(chunk) for chunk in result.chunks
                ],
                "citations": [chunk.citation for chunk in result.chunks],
            }
        except Exception as exc:
            return {
                "tool_name": "medical_rag",
                "tool_error": f"Medical RAG执行失败：{exc}",
            }

    def rag_synthesis_node(state: AgentState) -> AgentState:
        if state.get("tool_error"):
            return {
                "final_response": (
                    f"暂时无法检索医学资料：{state['tool_error']}。"
                    "请确认FAISS索引和Embedding配置。"
                )
            }
        context = str(state["tool_output"].get("context", ""))
        if not state.get("retrieved_evidence"):
            return {
                "final_response": (
                    "当前知识库未检索到足够资料，无法基于证据回答该问题。"
                    "建议核对知识库版本或由影像科医师查阅原始指南。"
                )
            }
        try:
            answer = model.answer_with_evidence(state["user_query"], context)
            return {"final_response": answer}
        except Exception as exc:
            return {
                "final_response": (
                    f"已检索到相关资料，但回答生成失败：{exc}\n\n{context}"
                ),
                "safety_warnings": ["Qwen证据摘要生成失败，返回原始检索资料"],
            }

    def unsupported_node(state: AgentState) -> AgentState:
        if state.get("intent") == "missing_features":
            answer = (
                "该问题需要当前病例的feature JSON。请先提供至少包含"
                "location、tumor_volume、enhancing_ratio和edema的分析结果。"
            )
        elif state.get("tool_error"):
            answer = str(state["tool_error"])
        else:
            answer = (
                "该问题超出医学影像辅助助手的受控范围。"
                "我可以总结MRI量化结果、生成影像辅助报告，"
                "或查询已入库的医学影像资料。"
            )
        return {"final_response": answer}

    def safety_node(state: AgentState) -> AgentState:
        response = state.get("final_response", "").strip()
        warnings = list(state.get("safety_warnings", []))
        violations = find_safety_violations(response)
        if violations:
            warnings.extend(violations)
            try:
                repaired = model.repair_safe_response(response, violations)
            except Exception:
                repaired = ""
            if repaired and not find_safety_violations(repaired):
                response = repaired
            else:
                response = (
                    "原回答包含超出医学影像辅助范围的诊断式表述，"
                    "已停止输出。建议由专业医师结合临床资料进行评估。"
                )
                warnings.append("安全改写失败，已使用本地安全响应")
        if SAFETY_NOTICE not in response:
            response = f"{response}\n\n{SAFETY_NOTICE}".strip()
        return {
            "final_response": response,
            "safety_warnings": warnings,
            "requires_human_review": True,
        }

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("mri_tool", mri_tool_node)
    graph.add_node("mri_synthesis", mri_synthesis_node)
    graph.add_node("report_tool", report_tool_node)
    graph.add_node("rag_tool", rag_tool_node)
    graph.add_node("rag_synthesis", rag_synthesis_node)
    graph.add_node("unsupported", unsupported_node)
    graph.add_node("safety", safety_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "mri_tool": "mri_tool",
            "report_tool": "report_tool",
            "rag_tool": "rag_tool",
            "unsupported": "unsupported",
        },
    )
    graph.add_edge("mri_tool", "mri_synthesis")
    graph.add_edge("mri_synthesis", "safety")
    graph.add_edge("report_tool", "safety")
    graph.add_edge("rag_tool", "rag_synthesis")
    graph.add_edge("rag_synthesis", "safety")
    graph.add_edge("unsupported", "safety")
    graph.add_edge("safety", END)
    return graph.compile()
