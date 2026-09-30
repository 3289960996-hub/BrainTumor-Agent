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

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    handlers = config.get("handlers")
    if isinstance(handlers, dict):
        for handler in handlers.values():
            if isinstance(handler, dict) and handler.get("filename"):
                Path(str(handler["filename"])).expanduser().parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
    logging.config.dictConfig(config)
