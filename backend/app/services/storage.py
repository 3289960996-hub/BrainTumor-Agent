"""基于本地文件系统的去标识化病例产物仓库。"""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.schemas.imaging import CASE_ID_PATTERN
from backend.app.services.errors import (
    AnalysisTaskNotFoundError,
    CaseConflictError,
    CaseNotFoundError,
    InvalidUploadError,
)

CASE_ID_REGEX = re.compile(CASE_ID_PATTERN)


@dataclass(frozen=True, slots=True)
class CasePaths:
    """单病例所有输入与衍生产物的固定路径。"""

    case_id: str
    root: Path
    raw: Path
    processed_root: Path
    processed_case: Path
    inference_root: Path
    mask: Path
    features: Path
    report: Path
    metadata: Path


class CaseRepository:
    """保存病例状态和产物，并确保用户输入不能逃逸数据根目录。"""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.cases_root = self.data_root / "cases"
        self.cases_root.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.Lock()
        self._case_locks: dict[str, threading.Lock] = {}

    def new_case_id(self) -> str:
        """生成不包含患者身份信息的服务端病例编号。"""

        return f"case-{uuid.uuid4().hex[:16]}"

    def validate_case_id(self, case_id: str) -> str:
        normalized = case_id.strip()
        if not CASE_ID_REGEX.fullmatch(normalized):
            raise InvalidUploadError(
                "case_id必须是1至64位字母、数字、下划线或连字符"
            )
        return normalized

    def paths(self, case_id: str) -> CasePaths:
        normalized = self.validate_case_id(case_id)
        root = (self.cases_root / normalized).resolve()
        if root.parent != self.cases_root:
            raise InvalidUploadError("case_id对应的存储路径不合法")
        return CasePaths(
            case_id=normalized,
            root=root,
            raw=root / "raw",
            processed_root=root / "processed",
            processed_case=root / "processed" / normalized,
            inference_root=root / "inference",
            mask=root / "inference" / "brats_predictions" / f"{normalized}.nii.gz",
            features=root / "features.json",
            report=root / "report.md",
            metadata=root / "case.json",
        )

    def create_case(self, case_id: str | None = None) -> CasePaths:
        resolved_id = self.validate_case_id(case_id) if case_id else self.new_case_id()
        paths = self.paths(resolved_id)
        if paths.root.exists():
            raise CaseConflictError(resolved_id)
        try:
            paths.raw.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise CaseConflictError(resolved_id) from exc
        self.write_status(
            resolved_id,
            "uploading",
            extra={"created_at": datetime.now(UTC).isoformat()},
        )
        return paths

    def require_case(self, case_id: str) -> CasePaths:
        paths = self.paths(case_id)
        if not paths.root.is_dir() or not paths.metadata.is_file():
            raise CaseNotFoundError(paths.case_id)
        return paths

    def write_status(
        self,
        case_id: str,
        status: str,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        paths = self.paths(case_id)
        current: dict[str, Any] = {}
        if paths.metadata.is_file():
            try:
                payload = json.loads(paths.metadata.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    current = payload
            except (OSError, json.JSONDecodeError):
                current = {}
        current.update(
            {
                "case_id": paths.case_id,
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        if extra:
            current.update(dict(extra))
        paths.root.mkdir(parents=True, exist_ok=True)
        temporary = paths.metadata.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(paths.metadata)

    def read_status(self, case_id: str) -> dict[str, Any]:
        paths = self.require_case(case_id)
        try:
            payload = json.loads(paths.metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaseNotFoundError(paths.case_id) from exc
        if not isinstance(payload, dict):
            raise CaseNotFoundError(paths.case_id)
        return payload

    def save_features(self, case_id: str, payload: Mapping[str, Any]) -> Path:
        paths = self.require_case(case_id)
        paths.features.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return paths.features

    def load_features(self, case_id: str) -> dict[str, Any] | None:
        paths = self.require_case(case_id)
        if not paths.features.is_file():
            return None
        payload = json.loads(paths.features.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return payload

    def save_report(self, case_id: str, report: str) -> Path:
        paths = self.require_case(case_id)
        paths.report.write_text(report.strip() + "\n", encoding="utf-8")
        return paths.report

    def save_report_revision(self, case_id: str, report: str) -> str:
        """保存旧报告副本并返回可审计的修订编号。"""

        paths = self.require_case(case_id)
        revision_id = f"rev-{uuid.uuid4().hex[:16]}"
        revisions = paths.root / "report_revisions"
        revisions.mkdir(parents=True, exist_ok=True)
        (revisions / f"{revision_id}.md").write_text(
            report.strip() + "\n", encoding="utf-8"
        )
        return revision_id

    def save_report_proposal(
        self,
        case_id: str,
        suggestion_id: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """持久化待医生确认的报告修改建议。"""

        paths = self.require_case(case_id)
        proposals = paths.root / "report_edit_proposals"
        proposals.mkdir(parents=True, exist_ok=True)
        target = proposals / f"{suggestion_id}.json"
        target.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    def load_report_proposal(
        self,
        case_id: str,
        suggestion_id: str,
    ) -> dict[str, Any]:
        """读取待确认建议，并限制路径在当前病例目录内。"""

        paths = self.require_case(case_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", suggestion_id):
            raise InvalidUploadError("报告修改建议编号格式不合法")
        target = paths.root / "report_edit_proposals" / f"{suggestion_id}.json"
        if not target.is_file():
            raise InvalidUploadError("报告修改建议不存在或已失效")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidUploadError("报告修改建议无法读取") from exc
        if not isinstance(payload, dict):
            raise InvalidUploadError("报告修改建议格式无效")
        return payload

    def list_cases(self, *, analyzed_only: bool = False) -> list[dict[str, Any]]:
        """Return de-identified case summaries without reading MRI payloads."""

        summaries: list[dict[str, Any]] = []
        if not self.cases_root.is_dir():
            return summaries
        for root in self.cases_root.iterdir():
            metadata = root / "case.json"
            if not root.is_dir() or not metadata.is_file():
                continue
            try:
                payload = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            case_id = payload.get("case_id")
            if not isinstance(case_id, str) or not CASE_ID_REGEX.fullmatch(case_id):
                continue
            has_metrics = (root / "features.json").is_file()
            if analyzed_only and not has_metrics:
                continue
            summaries.append(
                {
                    "case_id": case_id,
                    "status": str(payload.get("status", "uploaded")),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "has_metrics": has_metrics,
                    "has_report": (root / "report.md").is_file(),
                }
            )
        return sorted(
            summaries,
            key=lambda item: str(item.get("updated_at") or ""),
            reverse=True,
        )

    def remove_uncommitted_case(self, case_id: str) -> None:
        """仅清理刚创建但尚未完成上传的精确病例目录。"""

        paths = self.paths(case_id)
        if paths.root.parent != self.cases_root:
            raise InvalidUploadError("拒绝清理数据根目录以外的路径")
        if paths.root.is_dir():
            shutil.rmtree(paths.root)

    @contextmanager
    def case_lock(self, case_id: str) -> Iterator[None]:
        normalized = self.validate_case_id(case_id)
        with self._locks_guard:
            lock = self._case_locks.setdefault(normalized, threading.Lock())
        with lock:
            yield


class AnalysisTaskRepository:
    """Atomic JSON task records that survive API, worker, and browser restarts."""

    ACTIVE_STATUSES = {"queued", "running", "cancel_requested"}

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root).expanduser().resolve() / "analysis_tasks"
        self._lock = threading.Lock()

    def _path(self, task_id: str) -> Path:
        if not re.fullmatch(r"task-[a-f0-9]{32}", task_id):
            raise InvalidUploadError("分析任务编号格式不合法")
        target = (self.root / f"{task_id}.json").resolve()
        if target.parent != self.root:
            raise InvalidUploadError("分析任务路径不合法")
        return target

    def create(self, case_id: str) -> tuple[dict[str, Any], bool]:
        now = datetime.now(UTC).isoformat()
        payload: dict[str, Any] = {
            "task_id": f"task-{uuid.uuid4().hex}",
            "case_id": case_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "分析任务已进入队列",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "attempt": 0,
            "error_code": None,
            "error_message": None,
        }
        with self._lock:
            active = self.find_active_for_case(case_id, locked=True)
            if active is not None:
                return active, False
            self._write(payload)
        return payload, True

    def get(self, task_id: str) -> dict[str, Any]:
        target = self._path(task_id)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisTaskNotFoundError(task_id) from exc
        if not isinstance(payload, dict):
            raise AnalysisTaskNotFoundError(task_id)
        return payload

    def update(self, task_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            payload = self.get(task_id)
            payload.update(changes)
            payload["updated_at"] = datetime.now(UTC).isoformat()
            self._write(payload)
        return payload

    def latest_for_case(self, case_id: str) -> dict[str, Any] | None:
        matches = [
            payload
            for payload in self._all()
            if payload.get("case_id") == case_id
        ]
        return max(matches, key=lambda item: str(item.get("created_at", ""))) if matches else None

    def find_active_for_case(
        self,
        case_id: str,
        *,
        locked: bool = False,
    ) -> dict[str, Any] | None:
        def find() -> dict[str, Any] | None:
            latest = self.latest_for_case(case_id)
            if latest and latest.get("status") in self.ACTIVE_STATUSES:
                return latest
            return None

        if locked:
            return find()
        with self._lock:
            return find()

    def _all(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for target in self.root.glob("task-*.json"):
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(str(payload["task_id"]))
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
