"""MRI辅助报告模板与Qwen生成器测试。"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from report.generator import (
    QwenReportGenerator,
    ReportConfig,
    ReportGenerationError,
    load_analysis_json,
    save_report,
)
from report.template import MRIAnalysisInput, ReportInputError


class _FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        content = next(self.contents)
        return SimpleNamespace(
            id=f"request-{len(self.requests)}",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ],
        )


class _FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.completions = _FakeCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


def _safe_response() -> str:
    return json.dumps(
        {
            "imaging_summary": (
                "影像表现提示：AI自动分割区域主要位于左额叶，"
                "可见相应肿瘤区域及水肿分割信号。"
            ),
            "attention_items": [
                "建议结合临床表现及既往影像资料综合评估。",
                "建议由影像科医师复核AI分割边界。",
            ],
        },
        ensure_ascii=False,
    )


def test_input_validation_accepts_base_and_extended_json() -> None:
    analysis = MRIAnalysisInput.from_mapping(
        {
            "location": "left frontal",
            "tumor_volume": 35.5,
            "enhancing_ratio": 0.42,
            "edema": True,
            "max_diameter": 46.2,
            "enhancing_volume": 4.8,
        }
    )

    assert analysis.location == "left frontal"
    assert analysis.max_diameter == 46.2
    assert analysis.enhancing_volume == 4.8


def test_input_validation_rejects_invalid_ratio() -> None:
    with pytest.raises(ReportInputError, match="enhancing_ratio"):
        MRIAnalysisInput.from_mapping(
            {
                "location": "left frontal",
                "tumor_volume": 35.5,
                "enhancing_ratio": 1.2,
                "edema": True,
            }
        )


def test_generator_builds_five_section_safe_report() -> None:
    client = _FakeClient([_safe_response()])
    generator = QwenReportGenerator(
        config=ReportConfig(api_key="test-key"),
        client=client,
    )

    result = generator.generate(
        {
            "location": "left frontal",
            "tumor_volume": 35.5,
            "enhancing_ratio": 0.42,
            "edema": True,
        },
        case_id="BraTS2021_00001",
    )

    assert "## 1. 检查信息" in result.content
    assert "## 2. AI分割结果" in result.content
    assert "## 3. 肿瘤量化指标" in result.content
    assert "## 4. 影像表现总结" in result.content
    assert "## 5. 建议关注指标" in result.content
    assert "35.5 cm³" in result.content
    assert "42.00%" in result.content
    assert "影像表现提示" in result.content
    assert "建议结合临床" in result.content
    assert result.to_agent_payload()["requires_human_review"] is True
    request = client.completions.requests[0]
    assert request["model"] == "qwen-plus"
    assert "extra_headers" not in request


def test_data_inspection_header_is_explicitly_opt_in() -> None:
    client = _FakeClient([_safe_response()])
    generator = QwenReportGenerator(
        config=ReportConfig(
            api_key="test-key",
            enable_data_inspection=True,
        ),
        client=client,
    )

    generator.generate(
        {
            "location": "left frontal",
            "tumor_volume": 35.5,
            "enhancing_ratio": 0.42,
            "edema": True,
        }
    )

    request = client.completions.requests[0]
    assert "X-DashScope-DataInspection" in request["extra_headers"]


def test_unsafe_diagnostic_language_is_repaired() -> None:
    unsafe = json.dumps(
        {
            "imaging_summary": "影像表现提示：明确诊断为胶质母细胞瘤。",
            "attention_items": ["建议结合临床处理。"],
        },
        ensure_ascii=False,
    )
    client = _FakeClient([unsafe, _safe_response()])
    generator = QwenReportGenerator(
        config=ReportConfig(api_key="test-key", max_retries=1),
        client=client,
    )

    result = generator.generate(
        {
            "location": "left frontal",
            "tumor_volume": 35.5,
            "enhancing_ratio": 0.42,
            "edema": True,
        }
    )

    assert "明确诊断为" not in result.content
    assert len(client.completions.requests) == 2
    repair_prompt = client.completions.requests[1]["messages"][1]["content"]
    assert "上一次输出未通过安全校验" in repair_prompt


def test_repeated_unsafe_output_is_rejected() -> None:
    unsafe = json.dumps(
        {
            "imaging_summary": "影像表现提示：可确诊为脑膜瘤。",
            "attention_items": ["建议结合临床处理。"],
        },
        ensure_ascii=False,
    )
    generator = QwenReportGenerator(
        config=ReportConfig(api_key="test-key", max_retries=1),
        client=_FakeClient([unsafe, unsafe]),
    )

    with pytest.raises(ReportGenerationError, match="安全校验"):
        generator.generate(
            {
                "location": "right temporal",
                "tumor_volume": 10.0,
                "enhancing_ratio": 0.2,
                "edema": False,
            }
        )


def test_load_and_save_report(tmp_path: Path) -> None:
    input_path = tmp_path / "analysis.json"
    input_path.write_text(
        json.dumps(
            {
                "location": "left frontal",
                "tumor_volume": 35.5,
                "enhancing_ratio": 0.42,
                "edema": True,
            }
        ),
        encoding="utf-8",
    )
    analysis = load_analysis_json(input_path)
    generator = QwenReportGenerator(
        config=ReportConfig(api_key="test-key"),
        client=_FakeClient([_safe_response()]),
    )
    result = generator.generate(analysis)

    output_path = save_report(result, tmp_path / "report.md")

    assert output_path.is_file()
    assert "MRI影像辅助分析报告" in output_path.read_text(encoding="utf-8")
