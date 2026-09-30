"""日志配置加载：让服务日志与审计日志真正落到文件。"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_CONFIG = PROJECT_ROOT / "configs" / "logging.yaml"


def configure_logging(config_path: str | Path | None = None) -> None:
    """按``configs/logging.yaml``配置日志；配置文件缺失时退回基础配置。"""

    path = Path(config_path) if config_path is not None else DEFAULT_LOG_CONFIG
    if not path.is_file():
        logging.basicConfig(level=logging.INFO)
        return

    # utf-8-sig：用记事本等编辑器保存的配置文件可能带BOM，不能让它变成解析错误。
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    handlers = config.get("handlers")
    if isinstance(handlers, dict):
        for handler in handlers.values():
            if isinstance(handler, dict) and handler.get("filename"):
                Path(str(handler["filename"])).expanduser().parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
    logging.config.dictConfig(config)
