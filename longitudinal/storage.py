"""Persistent JSON storage for longitudinal comparisons."""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.services.errors import (
    ComparisonNotFoundError,
    ComparisonTaskNotFoundError,
    InvalidUploadError,
)

COMPARISON_ID_PATTERN = re.compile(r"comparison-[a-f0-9]{20}")
ARTIFACT_FILENAMES = {
    "registered_followup_t1ce": "followup_t1ce_registered.nii.gz",
    "registered_followup_mask": "followup_mask_registered.nii.gz",
    "rigid_transform": "rigid_transform.tfm",
    "wt_change": "wt_change.nii.gz",
    "tc_change": "tc_change.nii.gz",
    "et_change": "et_change.nii.gz",
}


class ComparisonRepository:
    """Store comparison records outside individual case directories."""

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root).expanduser().resolve() / "comparisons"

    def _path(self, comparison_id: str) -> Path:
        if not COMPARISON_ID_PATTERN.fullmatch(comparison_id):
            raise InvalidUploadError("随访对比编号格式不合法")
        target = (self.root / comparison_id / "comparison.json").resolve()
        if target.parent.parent != self.root:
            raise InvalidUploadError("随访对比路径不合法")
        return target

    def save(self, comparison_id: str, payload: Mapping[str, Any]) -> Path:
        target = self._path(comparison_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def artifact_dir(self, comparison_id: str) -> Path:
        return self._path(comparison_id).parent / "artifacts"

    def artifact(self, comparison_id: str, artifact_key: str) -> Path:
        filename = ARTIFACT_FILENAMES.get(artifact_key)
        if filename is None:
            raise InvalidUploadError("空间对比产物类型不合法")
        root = self.artifact_dir(comparison_id).resolve()
        target = (root / filename).resolve()
        if target.parent != root or not target.is_file():
            raise ComparisonNotFoundError(comparison_id)
        return target

    def get(self, comparison_id: str) -> dict[str, Any]:
        target = self._path(comparison_id)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ComparisonNotFoundError(comparison_id) from exc
        if not isinstance(payload, dict):
            raise ComparisonNotFoundError(comparison_id)
        return payload


class ComparisonTaskRepository:
    """Persistent task state for asynchronous longitudinal comparisons."""

    ACTIVE_STATUSES = {"queued", "running", "cancel_requested"}

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root).expanduser().resolve() / "comparison_tasks"
        self._lock = threading.Lock()

    def _path(self, task_id: str) -> Path:
        if not re.fullmatch(r"comparison-task-[a-f0-9]{32}", task_id):
            raise InvalidUploadError("空间对比任务编号格式不合法")
        target = (self.root / f"{task_id}.json").resolve()
        if target.parent != self.root:
            raise InvalidUploadError("空间对比任务路径不合法")
        return target

    def create(
        self,
        *,
        comparison_id: str,
        request_payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        now = datetime.now(UTC).isoformat()
        payload: dict[str, Any] = {
            "task_id": f"comparison-task-{uuid.uuid4().hex}",
            "comparison_id": comparison_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "空间对比任务已进入队列",
            "request": dict(request_payload),
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "attempt": 0,
            "error_code": None,
            "error_message": None,
        }
        with self._lock:
            active = self.find_active(comparison_id, locked=True)
            if active is not None:
                return active, False
            self._write(payload)
        return payload, True

    def get(self, task_id: str) -> dict[str, Any]:
        target = self._path(task_id)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ComparisonTaskNotFoundError(task_id) from exc
        if not isinstance(payload, dict):
            raise ComparisonTaskNotFoundError(task_id)
        return payload

    def update(self, task_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            payload = self.get(task_id)
            payload.update(changes)
            payload["updated_at"] = datetime.now(UTC).isoformat()
            self._write(payload)
        return payload

    def latest(self, comparison_id: str) -> dict[str, Any] | None:
        matches = [
            payload
            for payload in self._all()
            if payload.get("comparison_id") == comparison_id
        ]
        return (
            max(matches, key=lambda item: str(item.get("created_at", "")))
            if matches
            else None
        )

    def find_active(
        self,
        comparison_id: str,
        *,
        locked: bool = False,
    ) -> dict[str, Any] | None:
        def find() -> dict[str, Any] | None:
            latest = self.latest(comparison_id)
            if latest and latest.get("status") in self.ACTIVE_STATUSES:
                return latest
            return None

        if locked:
            return find()
        with self._lock:
            return find()

    def _all(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for target in self.root.glob("comparison-task-*.json"):
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
