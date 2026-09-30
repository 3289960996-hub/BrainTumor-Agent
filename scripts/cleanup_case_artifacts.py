"""按需清理失败或取消病例的中间产物（默认只演练，不删除任何内容）。

只有显式传入 ``--apply`` 才会删除病例下的 ``processed/`` 与 ``inference/``
目录；``raw/`` 原始影像、``case.json``、``features.json`` 与 ``report.md``
始终保留，方便重新分析或人工排查。

用法：

    python scripts/cleanup_case_artifacts.py                 # 演练：只列出将删除的目录
    python scripts/cleanup_case_artifacts.py --apply         # 真正删除
    python scripts/cleanup_case_artifacts.py --case-id case-xxx --apply
    python scripts/cleanup_case_artifacts.py --status analysis_failed --min-age-days 7
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_STATUSES = ("analysis_failed", "cancelled")
REMOVABLE_DIRS = ("processed", "inference")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清理失败病例的中间产物")
    parser.add_argument("--data-root", default=None, help="数据根目录；默认取BTA_DATA_ROOT")
    parser.add_argument(
        "--status",
        default=",".join(DEFAULT_STATUSES),
        help="需要清理的病例状态，逗号分隔",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="只处理指定病例，可重复传入",
    )
    parser.add_argument(
        "--min-age-days",
        type=float,
        default=0.0,
        help="只处理更新时间早于该天数的病例",
    )
    parser.add_argument("--apply", action="store_true", help="真正删除；缺省为演练模式")
    return parser.parse_args(argv)


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main(argv: list[str] | None = None) -> int:
    from backend.app.core.config import get_settings
    from backend.app.schemas.imaging import CASE_ID_PATTERN
    from backend.app.services.storage import CaseRepository

    args = _parse_args(argv)
    settings = get_settings()
    data_root = Path(args.data_root) if args.data_root else settings.data_root
    repository = CaseRepository(data_root)

    statuses = {item.strip() for item in args.status.split(",") if item.strip()}
    selected = set(args.case_id)
    cutoff = datetime.now(UTC) - timedelta(days=args.min_age_days)
    case_id_regex = re.compile(CASE_ID_PATTERN)

    hits = 0
    freed = 0
    for case_dir in sorted(repository.cases_root.iterdir()):
        if not case_dir.is_dir() or not case_id_regex.fullmatch(case_dir.name):
            continue
        case_id = case_dir.name
        if selected and case_id not in selected:
            continue
        paths = repository.paths(case_id)
        if not paths.metadata.is_file():
            continue
        try:
            payload = json.loads(paths.metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status", "")) not in statuses:
            continue
        if args.min_age_days:
            stamp = str(payload.get("updated_at") or payload.get("created_at") or "")
            try:
                if datetime.fromisoformat(stamp) > cutoff:
                    continue
            except ValueError:
                continue

        for name in REMOVABLE_DIRS:
            target = (paths.root / name).resolve()
            if target.parent != paths.root or not target.is_dir():
                continue
            size = _dir_size(target)
            if args.apply:
                shutil.rmtree(target)
            hits += 1
            freed += size
            action = "已删除" if args.apply else "将删除"
            print(f"{action} {case_id}/{name}  ({size / 1024 / 1024:.1f} MB)")

    mode = "应用模式" if args.apply else "演练模式（未删除任何内容，加 --apply 才执行）"
    print(f"\n{mode}：命中目录 {hits} 个，合计 {freed / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
