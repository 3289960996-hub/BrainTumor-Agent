"""四模态BraTS MRI上传服务。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
from fastapi import UploadFile

from backend.app.services.errors import InvalidUploadError
from backend.app.services.storage import CaseRepository
from data_process.constants import MRIModality


@dataclass(frozen=True, slots=True)
class UploadedCase:
    case_id: str
    modality_files: dict[str, str]


class MRIUploadService:
    """以流式方式保存四模态文件并执行轻量NIfTI头检查。"""

    def __init__(
        self,
        repository: CaseRepository,
        *,
        max_file_bytes: int,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> None:
        if max_file_bytes < 1 or chunk_size < 1:
            raise ValueError("上传大小和分块大小必须大于0")
        self.repository = repository
        self.max_file_bytes = max_file_bytes
        self.chunk_size = chunk_size

    async def upload_case(
        self,
        uploads: Mapping[MRIModality, UploadFile],
        case_id: str | None = None,
    ) -> UploadedCase:
        expected = set(MRIModality)
        if set(uploads) != expected:
            raise InvalidUploadError("必须同时上传T1、T1ce、T2和FLAIR四个模态")
        for modality, upload in uploads.items():
            inferred = _infer_modality_from_filename(upload.filename)
            if inferred is not None and inferred is not modality:
                raise InvalidUploadError(
                    f"文件{upload.filename}更像{inferred.value}模态，"
                    f"不能作为{modality.value}上传"
                )

        paths = self.repository.create_case(case_id)
        saved: dict[str, str] = {}
        sizes: dict[str, int] = {}
        try:
            for modality in MRIModality:
                upload = uploads[modality]
                suffix = _nifti_suffix(upload.filename)
                target = paths.raw / f"{paths.case_id}_{modality.value}{suffix}"
                sizes[modality.value] = await self._save_file(upload, target)
                _validate_nifti_header(target)
                saved[modality.value] = target.name
            self.repository.write_status(
                paths.case_id,
                "uploaded",
                extra={
                    "modalities": saved,
                    "file_sizes_bytes": sizes,
                },
            )
        except Exception:
            self.repository.remove_uncommitted_case(paths.case_id)
            raise
        return UploadedCase(case_id=paths.case_id, modality_files=saved)

    async def _save_file(self, upload: UploadFile, target: Path) -> int:
        total = 0
        try:
            with target.open("xb") as output:
                while chunk := await upload.read(self.chunk_size):
                    total += len(chunk)
                    if total > self.max_file_bytes:
                        raise InvalidUploadError(
                            f"单个MRI文件不能超过{self.max_file_bytes}字节"
                        )
                    output.write(chunk)
        finally:
            await upload.close()
        if total == 0:
            raise InvalidUploadError("MRI文件不能为空")
        return total


def _nifti_suffix(filename: str | None) -> str:
    name = (filename or "").lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    raise InvalidUploadError("MRI文件仅支持.nii或.nii.gz格式")


def _infer_modality_from_filename(filename: str | None) -> MRIModality | None:
    name = (filename or "").lower()
    name = re.sub(r"\.nii(?:\.gz)?$", "", name)
    stem = re.sub(r"[^a-z0-9]+", "_", name)

    def has_token(pattern: str) -> bool:
        return re.search(rf"(?:^|_){pattern}(?:_|$)", stem) is not None

    if has_token("flair"):
        return MRIModality.FLAIR
    if has_token(r"t1(?:ce|c|gd|post)") or re.search(
        r"(?:^|_)t1_(?:ce|c|gd|post)(?:_|$)", stem
    ):
        return MRIModality.T1CE
    if has_token("t2"):
        return MRIModality.T2
    if has_token("t1"):
        return MRIModality.T1
    return None


def _validate_nifti_header(path: Path) -> None:
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise InvalidUploadError(f"无法读取NIfTI头信息：{path.name}") from exc
    if len(image.shape) != 3 or any(int(length) < 1 for length in image.shape):
        raise InvalidUploadError(f"MRI必须是非空三维NIfTI：{path.name}")
