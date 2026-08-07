"""Qwen-plus在MRI Assistant中的路由、摘要和证据问答适配器。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from agent.state import AgentIntent
from report.generator import DEFAULT_QWEN_BASE_URL


class AgentModelError(RuntimeError):
    """Qwen-plus调用或输出解析失败。"""


class _CompletionsAPI(Protocol):
    def create(self, **kwargs: Any) -> Any:
        """创建聊天完成。"""


class _ChatAPI(Protocol):
    completions: _CompletionsAPI


class OpenAICompatibleClient(Protocol):
    chat: _ChatAPI


@dataclass(frozen=True, slots=True)
class QwenAgentConfig:
    """Qwen-plus Agent配置。"""

    api_key: str
    base_url: str = DEFAULT_QWEN_BASE_URL
    model: str = "qwen-plus"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout_seconds: float = 60.0
    enable_data_inspection: bool = False

    @classmethod
    def from_env(cls) -> QwenAgentConfig:
        """从环境变量读取模型设置。"""

        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            base_url=os.getenv("BTA_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).strip(),
            model=os.getenv("BTA_QWEN_MODEL", "qwen-plus").strip(),
            temperature=float(os.getenv("BTA_AGENT_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("BTA_AGENT_MAX_TOKENS", "1000")),
            timeout_seconds=float(os.getenv("BTA_QWEN_TIMEOUT_SECONDS", "60")),
            enable_data_inspection=os.getenv(
                "BTA_QWEN_ENABLE_DATA_INSPECTION",
                "false",
            ).strip().lower()
            in {"1", "true", "yes", "on"},
        )


class AgentLanguageModel(Protocol):
    """LangGraph工作流依赖的最小模型协议。"""

    def classify_intent(
        self,
        user_query: str,
        has_features: bool,
    ) -> AgentIntent:
        """将模糊问题分类到白名单意图。"""

    def summarize_imaging(
        self,
        user_query: str,
        tool_output: Mapping[str, Any],
    ) -> str:
        """根据MRI工具结果生成影像摘要。"""

    def answer_with_evidence(
        self,
        user_query: str,
        evidence_context: str,
    ) -> str:
        """只依据RAG证据回答医学知识问题。"""

    def repair_safe_response(
        self,
        unsafe_response: str,
        violations: Sequence[str],
    ) -> str:
        """修复不符合辅助助手边界的回答。"""


class QwenAgentClient:
    """通过OpenAI兼容接口调用Qwen-plus。"""

    def __init__(
        self,
        config: QwenAgentConfig | None = None,
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self.config = config or QwenAgentConfig.from_env()
        _validate_config(self.config, require_api_key=client is None)
        self.client = client or self._create_client()

    def _create_client(self) -> OpenAICompatibleClient:
        """创建阿里云百炼OpenAI兼容客户端。"""

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentModelError("未安装openai，请先安装requirements.txt") from exc
        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=2,
        )

    def classify_intent(
        self,
        user_query: str,
        has_features: bool,
    ) -> AgentIntent:
        """让Qwen仅在五个受控意图之间选择。"""

        content = self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "你是医学影像辅助助手的意图分类器，只输出JSON。"
                        "允许意图：mri_summary、report_generation、"
                        "medical_knowledge、missing_features、unsupported。"
                        "禁止进行疾病诊断。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": user_query,
                            "has_mri_feature_json": has_features,
                            "rules": {
                                "mri_summary": "解释或总结当前MRI分析JSON",
                                "report_generation": "生成当前病例辅助报告",
                                "medical_knowledge": "查询指南、标准或影像学意义",
                                "missing_features": "问题需要病例指标但未提供JSON",
                                "unsupported": "超出医学影像辅助范围",
                            },
                            "output": {"intent": "one allowed intent"},
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=120,
        )
        payload = _parse_json_object(content)
        intent = payload.get("intent")
        allowed: set[str] = {
            "mri_summary",
            "report_generation",
            "medical_knowledge",
            "missing_features",
            "unsupported",
        }
        if intent not in allowed:
            raise AgentModelError(f"Qwen返回未知Agent意图：{intent!r}")
        return cast(AgentIntent, intent)

    def summarize_imaging(
        self,
        user_query: str,
        tool_output: Mapping[str, Any],
    ) -> str:
        """基于MRI工具的确定性解释生成简洁摘要。"""

        return self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "你是医学影像辅助助手。只能总结给定AI量化指标，"
                        "不得确诊疾病、推断病理类型或WHO分级。"
                        "使用“AI定量分析显示”或“影像表现提示”等审慎表达，"
                        "最后明确建议结合临床并由影像科医师审核。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"医生问题：{user_query}\n"
                        "MRI Analyzer输出：\n"
                        f"{json.dumps(dict(tool_output), ensure_ascii=False, indent=2)}"
                    ),
                },
            ]
        )

    def answer_with_evidence(
        self,
        user_query: str,
        evidence_context: str,
    ) -> str:
        """仅根据Retriever资料回答，并保留资料编号。"""

        return self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "你是医学影像知识辅助助手。只能依据给定资料回答，"
                        "关键陈述后标注[资料1]等引用。资料不足时明确说明。"
                        "不得给出患者疾病确诊或替代临床决策，"
                        "应建议结合临床和影像科医师判断。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"医生问题：{user_query}\n"
                        f"<retrieved_evidence>\n{evidence_context}\n"
                        "</retrieved_evidence>"
                    ),
                },
            ]
        )

    def repair_safe_response(
        self,
        unsafe_response: str,
        violations: Sequence[str],
    ) -> str:
        """将越界诊断措辞改写为影像辅助表达。"""

        return self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "将回答改写为医学影像辅助表达。删除疾病确诊、病理类型、"
                        "WHO分级和治疗决策；保留已有量化数据和资料引用。"
                        "使用“影像表现提示”“建议结合临床”。只输出改写正文。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"安全问题：{'；'.join(violations)}\n"
                        f"待改写回答：\n{unsafe_response}"
                    ),
                },
            ],
            temperature=0.0,
        )

    def _complete(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """执行一次非流式聊天完成。"""

        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if self.config.enable_data_inspection:
            request["extra_headers"] = {
                "X-DashScope-DataInspection": '{"input":"cip","output":"cip"}'
            }
        try:
            response = self.client.chat.completions.create(**request)
            content = response.choices[0].message.content
        except Exception as exc:
            raise AgentModelError(f"Qwen-plus API调用失败：{exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise AgentModelError("Qwen-plus返回了空回答")
        return content.strip()


def _validate_config(config: QwenAgentConfig, require_api_key: bool) -> None:
    """验证Qwen Agent配置。"""

    if require_api_key and not config.api_key:
        raise AgentModelError("未配置DASHSCOPE_API_KEY")
    if not config.base_url.startswith(("https://", "http://")):
        raise AgentModelError("BTA_QWEN_BASE_URL必须是HTTP(S)地址")
    if not config.model:
        raise AgentModelError("BTA_QWEN_MODEL不能为空")
    if not 0.0 <= config.temperature <= 2.0:
        raise AgentModelError("BTA_AGENT_TEMPERATURE必须位于0到2之间")
    if config.max_tokens < 100:
        raise AgentModelError("BTA_AGENT_MAX_TOKENS不能小于100")


def _parse_json_object(content: str) -> dict[str, Any]:
    """解析Qwen可能使用Markdown围栏包装的JSON。"""

    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AgentModelError("Qwen意图分类输出不是合法JSON") from exc
    if not isinstance(payload, dict):
        raise AgentModelError("Qwen意图分类输出必须是JSON object")
    return payload
