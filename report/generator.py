"""调用Qwen-plus生成MRI影像辅助报告。"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from report.template import (
    MRIAnalysisInput,
    ReportInputError,
    build_narrative_messages,
    render_report,
)

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 只拦截直接确诊式表达；审慎的“影像表现提示”不会命中。
FORBIDDEN_DIAGNOSIS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:明确|最终|临床)?诊断为"),
    re.compile(r"(?:可以|可|已经|已)?确诊为?"),
    re.compile(r"考虑为(?:胶质|脑膜|转移|淋巴|生殖细胞|室管膜)"),
    re.compile(r"符合.{0,20}(?:疾病|肿瘤).{0,10}诊断"),
    re.compile(r"WHO\s*[ⅠIⅡVⅢIV1-4]+级", re.IGNORECASE),
)


class _CompletionsAPI(Protocol):
    """OpenAI兼容chat.completions接口的最小协议。"""

    def create(self, **kwargs: Any) -> Any:
        """创建一次非流式聊天完成。"""


class _ChatAPI(Protocol):
    completions: _CompletionsAPI


class _OpenAICompatibleClient(Protocol):
    chat: _ChatAPI


class ReportGenerationError(RuntimeError):
    """Qwen调用、输出解析或医学安全校验失败。"""


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """Qwen-plus报告生成配置。"""

    api_key: str
    base_url: str = DEFAULT_QWEN_BASE_URL
    model: str = "qwen-plus"
    temperature: float = 0.2
    max_tokens: int = 800
    timeout_seconds: float = 60.0
    max_retries: int = 1
    enable_data_inspection: bool = True

    @classmethod
    def from_env(cls) -> ReportConfig:
        """从环境变量读取配置，不在代码或日志中暴露API Key。"""

        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            base_url=os.getenv("BTA_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).strip(),
            model=os.getenv("BTA_QWEN_MODEL", "qwen-plus").strip(),
            temperature=float(os.getenv("BTA_REPORT_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("BTA_REPORT_MAX_TOKENS", "800")),
            timeout_seconds=float(os.getenv("BTA_QWEN_TIMEOUT_SECONDS", "60")),
        )


@dataclass(frozen=True, slots=True)
class GeneratedReport:
    """报告正文及调用审计所需的非敏感元数据。"""

    content: str
    model: str
    request_id: str | None
    analysis: MRIAnalysisInput

    def to_agent_payload(self) -> dict[str, Any]:
        """转换为后续Agent状态可直接使用的字典。"""

        return {
            "report": self.content,
            "model": self.model,
            "request_id": self.request_id,
            "analysis": self.analysis.to_prompt_payload(),
            "requires_human_review": True,
        }


@dataclass(frozen=True, slots=True)
class _Narrative:
    imaging_summary: str
    attention_items: tuple[str, ...]


class QwenReportGenerator:
    """通过OpenAI兼容接口调用Qwen-plus，并执行医学安全后校验。"""

    def __init__(
        self,
        config: ReportConfig | None = None,
        client: _OpenAICompatibleClient | None = None,
    ) -> None:
        self.config = config or ReportConfig.from_env()
        _validate_config(self.config, require_api_key=client is None)
        self.client = client or self._create_client()

    def _create_client(self) -> _OpenAICompatibleClient:
        """延迟导入OpenAI SDK，方便本地模板和测试在无网络环境运行。"""

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ReportGenerationError(
                "未安装openai，请先执行pip install -r requirements.txt"
            ) from exc

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=2,
        )

    def generate(
        self,
        analysis: MRIAnalysisInput | Mapping[str, Any],
        case_id: str | None = None,
    ) -> GeneratedReport:
        """生成完整五章节辅助报告，失败时最多进行一次安全修复。"""

        normalized = (
            analysis
            if isinstance(analysis, MRIAnalysisInput)
            else MRIAnalysisInput.from_mapping(analysis)
        )
        feedback: tuple[str, ...] = ()
        last_error: ReportGenerationError | None = None

        for _ in range(self.config.max_retries + 1):
            try:
                response = self._request_narrative(normalized, feedback)
                narrative = _parse_narrative(_extract_content(response))
                violations = validate_narrative(narrative)
                if violations:
                    raise ReportGenerationError("；".join(violations))
                report = render_report(
                    analysis=normalized,
                    imaging_summary=narrative.imaging_summary,
                    attention_items=narrative.attention_items,
                    case_id=case_id,
                )
                return GeneratedReport(
                    content=report,
                    model=self.config.model,
                    request_id=_extract_request_id(response),
                    analysis=normalized,
                )
            except ReportGenerationError as exc:
                last_error = exc
                feedback = (str(exc),)

        raise ReportGenerationError(
            f"Qwen-plus输出在修复后仍未通过报告安全校验：{last_error}"
        ) from last_error

    def _request_narrative(
        self,
        analysis: MRIAnalysisInput,
        feedback: Sequence[str],
    ) -> Any:
        """发起非流式Qwen-plus调用。"""

        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": build_narrative_messages(analysis, feedback),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.enable_data_inspection:
            request["extra_headers"] = {
                "X-DashScope-DataInspection": '{"input":"cip","output":"cip"}'
            }
        try:
            return self.client.chat.completions.create(**request)
        except Exception as exc:
            raise ReportGenerationError(f"Qwen-plus API调用失败：{exc}") from exc


def _validate_config(config: ReportConfig, require_api_key: bool) -> None:
    """在发起请求前验证配置。"""

    if require_api_key and not config.api_key:
        raise ReportGenerationError("未配置DASHSCOPE_API_KEY")
    if not config.base_url.startswith(("https://", "http://")):
        raise ReportGenerationError("BTA_QWEN_BASE_URL必须是HTTP(S)地址")
    if not config.model:
        raise ReportGenerationError("BTA_QWEN_MODEL不能为空")
    if not 0.0 <= config.temperature <= 2.0:
        raise ReportGenerationError("temperature必须位于0到2之间")
    if config.max_tokens < 100:
        raise ReportGenerationError("max_tokens不能小于100")
    if config.max_retries < 0:
        raise ReportGenerationError("max_retries不能小于0")


def _extract_content(response: Any) -> str:
    """从OpenAI兼容响应中提取文本。"""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ReportGenerationError("Qwen-plus响应缺少choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ReportGenerationError("Qwen-plus返回了空内容")
    return content.strip()


def _extract_request_id(response: Any) -> str | None:
    """提取请求ID，供审计和问题排查使用。"""

    request_id = getattr(response, "id", None)
    return str(request_id) if request_id is not None else None


def _parse_narrative(content: str) -> _Narrative:
    """解析模型JSON，并拒绝夹带的自由文本。"""

    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:].lstrip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ReportGenerationError("模型输出不是合法JSON") from exc
    if not isinstance(payload, dict):
        raise ReportGenerationError("模型输出必须是JSON object")

    summary = payload.get("imaging_summary")
    items = payload.get("attention_items")
    if not isinstance(summary, str) or not summary.strip():
        raise ReportGenerationError("imaging_summary必须是非空字符串")
    if (
        not isinstance(items, list)
        or not items
        or not all(isinstance(item, str) and item.strip() for item in items)
    ):
        raise ReportGenerationError("attention_items必须是非空字符串数组")
    if len(items) > 8:
        raise ReportGenerationError("attention_items不能超过8项")
    return _Narrative(
        imaging_summary=summary.strip(),
        attention_items=tuple(item.strip() for item in items),
    )


def validate_narrative(narrative: _Narrative) -> tuple[str, ...]:
    """检查确诊式表达、必需审慎措辞和异常长度。"""

    text = "\n".join((narrative.imaging_summary, *narrative.attention_items))
    violations: list[str] = []
    if "影像表现提示" not in narrative.imaging_summary:
        violations.append("影像表现总结必须包含“影像表现提示”")
    if not any("建议结合临床" in item for item in narrative.attention_items):
        violations.append("建议中必须包含“建议结合临床”")
    if len(narrative.imaging_summary) > 600:
        violations.append("影像表现总结过长")
    for pattern in FORBIDDEN_DIAGNOSIS_PATTERNS:
        if pattern.search(text):
            violations.append(f"包含禁止的直接诊断表达：{pattern.pattern}")
    return tuple(violations)


def load_analysis_json(path: str | Path) -> MRIAnalysisInput:
    """读取并校验MRI分析JSON。"""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReportInputError(f"MRI分析JSON不存在：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"无法读取MRI分析JSON：{source}") from exc
    if not isinstance(payload, dict):
        raise ReportInputError("MRI分析JSON根节点必须是object")
    return MRIAnalysisInput.from_mapping(payload)


def save_report(report: GeneratedReport, output_path: str | Path) -> Path:
    """保存UTF-8 Markdown辅助报告。"""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.content, encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    """创建报告生成命令行参数。"""

    parser = argparse.ArgumentParser(description="调用Qwen-plus生成MRI影像辅助报告。")
    parser.add_argument("--input-json", required=True, help="MRI分析JSON路径")
    parser.add_argument("--output", required=True, help="报告Markdown输出路径")
    parser.add_argument("--case-id", default=None, help="可选去标识化病例编号")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """加载分析JSON、调用Qwen-plus并保存报告。"""

    args = build_parser().parse_args(argv)
    try:
        analysis = load_analysis_json(args.input_json)
        generator = QwenReportGenerator()
        report = generator.generate(analysis, case_id=args.case_id)
        output_path = save_report(report, args.output)
    except (ReportInputError, ReportGenerationError, OSError) as exc:
        print(f"报告生成失败：{exc}")
        return 1

    print(f"辅助报告已生成：{output_path}")
    print(f"模型：{report.model}")
    if report.request_id:
        print(f"请求ID：{report.request_id}")
    print("注意：报告须由影像科医师审核，不可作为独立诊断依据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
