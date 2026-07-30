"""医学PDF加载、清洗、切分和可追溯元数据管理。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ALLOWED_TOPICS = {
    "who_cns",
    "nccn_glioma",
    "glioma_mri",
    "follow_up_criteria",
}
DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")


class KnowledgeLoadError(ValueError):
    """知识文档、manifest或PDF内容不符合入库要求。"""


class _PDFLoaderProtocol(Protocol):
    def load(self) -> list[Document]:
        """按页加载PDF。"""


PDFLoaderFactory = Callable[[Path], _PDFLoaderProtocol]


@dataclass(frozen=True, slots=True)
class PDFDocumentSpec:
    """一个获得授权、可入库PDF的版本化说明。"""

    file: str
    document_id: str
    title: str
    source: str
    version: str
    topic: str
    publication_date: str | None = None
    source_url: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PDFDocumentSpec:
        """从manifest条目构造并校验文档说明。"""

        required = ("file", "document_id", "title", "source", "version", "topic")
        missing = [key for key in required if key not in data]
        if missing:
            raise KnowledgeLoadError(f"manifest文档条目缺少：{', '.join(missing)}")

        values: dict[str, str] = {}
        for key in required:
            value = data[key]
            if not isinstance(value, str) or not value.strip():
                raise KnowledgeLoadError(f"manifest字段{key}必须是非空字符串")
            values[key] = value.strip()

        if not DOCUMENT_ID_PATTERN.fullmatch(values["document_id"]):
            raise KnowledgeLoadError(
                "document_id仅允许字母、数字、点、下划线和连字符，长度2至128"
            )
        if values["topic"] not in ALLOWED_TOPICS:
            raise KnowledgeLoadError(
                f"topic必须是以下之一：{', '.join(sorted(ALLOWED_TOPICS))}"
            )
        if not values["file"].lower().endswith(".pdf"):
            raise KnowledgeLoadError(f"知识文档必须是PDF：{values['file']}")

        extra_metadata = data.get("metadata", {})
        if not isinstance(extra_metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in extra_metadata.items()
        ):
            raise KnowledgeLoadError("metadata必须是字符串到字符串的JSON object")

        return cls(
            **values,
            publication_date=_optional_string(data, "publication_date"),
            source_url=_optional_string(data, "source_url"),
            metadata=dict(extra_metadata),
        )


def _optional_string(data: Mapping[str, Any], key: str) -> str | None:
    """读取可选非空字符串。"""

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeLoadError(f"{key}必须是非空字符串或null")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """医学资料文本切分配置，长度单位为Unicode字符。"""

    chunk_size: int = 900
    chunk_overlap: int = 150
    minimum_chunk_length: int = 80

    def validate(self) -> None:
        """拒绝会产生空块或过度重叠的参数。"""

        if self.chunk_size < 200:
            raise KnowledgeLoadError("chunk_size不能小于200")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise KnowledgeLoadError("chunk_overlap必须大于等于0且小于chunk_size")
        if self.minimum_chunk_length < 1:
            raise KnowledgeLoadError("minimum_chunk_length必须大于等于1")


def load_manifest(path: str | Path) -> tuple[PDFDocumentSpec, ...]:
    """读取知识库manifest并检查document_id和文件名不重复。"""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise KnowledgeLoadError(f"知识库manifest不存在：{manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeLoadError(f"无法读取知识库manifest：{manifest_path}") from exc

    raw_documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(raw_documents, list) or not raw_documents:
        raise KnowledgeLoadError("manifest必须包含非空documents数组")

    specs = tuple(PDFDocumentSpec.from_mapping(item) for item in raw_documents)
    _ensure_unique([spec.document_id for spec in specs], "document_id")
    _ensure_unique([spec.file.casefold() for spec in specs], "PDF文件")
    return specs


def _ensure_unique(values: Sequence[str], label: str) -> None:
    """检查manifest中的唯一字段。"""

    if len(set(values)) != len(values):
        raise KnowledgeLoadError(f"manifest包含重复{label}")


def clean_pdf_text(text: str) -> str:
    """规范Unicode和空白，同时尽量保留段落边界。"""

    normalized = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


class MedicalPDFLoader:
    """加载manifest声明的PDF并生成带页码引用的LangChain文档块。"""

    def __init__(
        self,
        pdf_root: str | Path,
        chunking: ChunkingConfig | None = None,
        pdf_loader_factory: PDFLoaderFactory | None = None,
    ) -> None:
        self.pdf_root = Path(pdf_root).expanduser().resolve()
        if not self.pdf_root.is_dir():
            raise KnowledgeLoadError(f"PDF根目录不存在：{self.pdf_root}")
        self.chunking = chunking or ChunkingConfig()
        self.chunking.validate()
        self.pdf_loader_factory = pdf_loader_factory or _default_pdf_loader
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking.chunk_size,
            chunk_overlap=self.chunking.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", ". ", "; ", ", ", " ", ""],
            keep_separator=True,
            add_start_index=True,
        )

    def load_manifest(self, manifest_path: str | Path) -> list[Document]:
        """加载manifest中的全部PDF并返回切分后的文档块。"""

        chunks: list[Document] = []
        for spec in load_manifest(manifest_path):
            chunks.extend(self.load_document(spec))
        if not chunks:
            raise KnowledgeLoadError("没有生成可入库的医学文本块")
        return chunks

    def load_document(self, spec: PDFDocumentSpec) -> list[Document]:
        """加载单个PDF、清洗页面文本并切分。"""

        pdf_path = _resolve_pdf_path(self.pdf_root, spec.file)
        if not pdf_path.is_file():
            raise KnowledgeLoadError(f"manifest声明的PDF不存在：{pdf_path}")
        checksum = _sha256_file(pdf_path)

        try:
            pages = self.pdf_loader_factory(pdf_path).load()
        except Exception as exc:
            raise KnowledgeLoadError(f"PDF解析失败：{pdf_path.name}：{exc}") from exc
        if not pages:
            raise KnowledgeLoadError(f"PDF未解析出页面：{pdf_path.name}")

        cleaned_pages: list[Document] = []
        for fallback_page, page in enumerate(pages, start=1):
            text = clean_pdf_text(page.page_content)
            if not text:
                continue
            page_index = page.metadata.get("page", fallback_page - 1)
            page_number = int(page_index) + 1
            metadata: dict[str, Any] = {
                "document_id": spec.document_id,
                "title": spec.title,
                "source": spec.source,
                "version": spec.version,
                "topic": spec.topic,
                "file_name": pdf_path.name,
                "file_sha256": checksum,
                "page": page_number,
                "citation": f"{spec.title} ({spec.version}), p.{page_number}",
                **spec.metadata,
            }
            if spec.publication_date:
                metadata["publication_date"] = spec.publication_date
            if spec.source_url:
                metadata["source_url"] = spec.source_url
            cleaned_pages.append(Document(page_content=text, metadata=metadata))

        if not cleaned_pages:
            raise KnowledgeLoadError(
                f"PDF页面均无可提取文本，可能是扫描件，需要先执行OCR：{pdf_path.name}"
            )

        chunks = self.splitter.split_documents(cleaned_pages)
        retained: list[Document] = []
        for page_chunk_index, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if len(content) < self.chunking.minimum_chunk_length:
                continue
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = page_chunk_index
            metadata["chunk_id"] = _chunk_id(metadata, content)
            retained.append(Document(page_content=content, metadata=metadata))
        if not retained:
            raise KnowledgeLoadError(f"切分后没有达到最小长度的文本块：{pdf_path.name}")
        return retained


def _resolve_pdf_path(pdf_root: Path, relative_path: str) -> Path:
    """限制manifest只能引用PDF根目录内部文件。"""

    candidate = (pdf_root / relative_path).resolve()
    try:
        candidate.relative_to(pdf_root)
    except ValueError as exc:
        raise KnowledgeLoadError(f"PDF路径越出知识库根目录：{relative_path}") from exc
    return candidate


def _default_pdf_loader(path: Path) -> _PDFLoaderProtocol:
    """延迟创建LangChain PyPDFLoader。"""

    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as exc:
        raise KnowledgeLoadError(
            "缺少PDF依赖，请安装langchain-community和pypdf"
        ) from exc
    return PyPDFLoader(str(path))


def _sha256_file(path: Path) -> str:
    """流式计算文件SHA-256，避免整份指南加载到内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_id(metadata: Mapping[str, Any], content: str) -> str:
    """根据文档、页码、起点和正文生成稳定chunk ID。"""

    identity = "|".join(
        (
            str(metadata["document_id"]),
            str(metadata["page"]),
            str(metadata.get("start_index", 0)),
            content,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
