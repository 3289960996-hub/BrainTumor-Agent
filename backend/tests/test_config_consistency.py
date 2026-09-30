"""配置一致性测试：`.env.example`、`configs/app.yaml` 与代码默认值不能漂移。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from backend.app.core.config import Settings, get_settings
from rag.embedding import BGEEmbeddingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_YAML = PROJECT_ROOT / "configs" / "app.yaml"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

# configs/app.yaml 中与Settings重叠、必须保持一致的键。
APP_YAML_MAPPING: dict[tuple[str, ...], str] = {
    ("application", "version"): "app_version",
    ("application", "environment"): "environment",
    ("application", "api_prefix"): "api_prefix",
    ("storage", "data_root"): "data_root",
    ("storage", "model_root"): "model_root",
    ("storage", "faiss_index_path"): "faiss_index_path",
    ("storage", "max_upload_size_mb"): "max_upload_size_mb",
    ("segmentation", "dataset_id"): "nnunet_dataset_id",
    ("segmentation", "configuration"): "nnunet_configuration",
    ("segmentation", "plans"): "nnunet_plans",
    ("segmentation", "trainer"): "nnunet_trainer",
    ("segmentation", "folds"): "nnunet_folds",
    ("segmentation", "device"): "nnunet_device",
    ("segmentation", "gpu_id"): "nnunet_gpu_id",
    ("segmentation", "checkpoint"): "nnunet_checkpoint",
    ("rag", "embedding_model"): "embedding_model",
    ("agent", "model"): "qwen_model",
    ("agent", "temperature"): "agent_temperature",
    ("agent", "max_tokens"): "agent_max_tokens",
    ("report", "temperature"): "report_temperature",
    ("report", "max_tokens"): "report_max_tokens",
}


def _nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        current = current[key]
    return current


def _field_default(name: str) -> Any:
    """取Settings的字段默认值；不受开发者本机.env覆盖的影响。"""

    field = Settings.model_fields[name]
    if field.default_factory is not None:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    return field.default


def test_app_yaml_documented_values_match_settings_defaults() -> None:
    """configs/app.yaml声明的默认值必须与代码默认值一致。"""

    payload = yaml.safe_load(APP_YAML.read_text(encoding="utf-8-sig"))
    mismatched: list[str] = []
    for keys, field_name in APP_YAML_MAPPING.items():
        documented = _nested(payload, keys)
        default = _field_default(field_name)
        if isinstance(default, Path):
            matches = Path(str(documented)) == default
        elif isinstance(default, list):
            matches = [str(item) for item in documented] == [str(item) for item in default]
        else:
            matches = documented == default
        if not matches:
            mismatched.append(f"{'.'.join(keys)}={documented!r} != {field_name}默认值{default!r}")
    assert not mismatched, "configs/app.yaml与代码默认值不一致：" + "；".join(mismatched)


def test_env_example_keys_are_known_to_settings() -> None:
    """`.env.example`里每个变量都要能被Settings读到，不能静默忽略。"""

    keys = [
        line.split("=", 1)[0].strip()
        for line in ENV_EXAMPLE.read_text(encoding="utf-8-sig").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    ]
    accepted = {f"BTA_{name.upper()}" for name in Settings.model_fields}
    accepted |= {"NNUNET_ROOT", "BTA_NNUNET_ROOT", "DASHSCOPE_API_KEY"}

    unknown = sorted({key for key in keys if key not in accepted})
    assert not unknown, f".env.example中这些变量Settings不认识，会被静默忽略：{unknown}"


def test_embedding_config_follows_dotenv_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """embedding配置现在走Settings，写在.env里的值会真正生效。"""

    monkeypatch.setenv("BTA_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("BTA_EMBEDDING_BATCH_SIZE", "4")
    get_settings.cache_clear()
    try:
        config = BGEEmbeddingConfig.from_env()
    finally:
        get_settings.cache_clear()

    assert config.device == "cpu"
    assert config.batch_size == 4


@pytest.mark.parametrize("variable", ["BTA_NNUNET_ROOT", "NNUNET_ROOT"])
def test_nnunet_root_accepts_both_variable_names(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    """NNUNET_ROOT 与 BTA_NNUNET_ROOT 都应生效。"""

    for name in ("BTA_NNUNET_ROOT", "NNUNET_ROOT"):
        monkeypatch.delenv(name, raising=False)
    expected = Path(f"./runtime/{variable.lower()}")
    monkeypatch.setenv(variable, str(expected))
    get_settings.cache_clear()
    try:
        resolved = get_settings().nnunet_root
    finally:
        get_settings.cache_clear()

    assert resolved == expected
