"""FAISS医学知识索引的构建、保存、校验和加载。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.embedding import (
    BGEEmbeddingConfig,
    EmbeddingSetupError,
    create_bge_embeddings,
)

INDEX_MANIFEST_FILE = "index_manifest.json"
FAISS_INDEX_FILE = "index.faiss"
FAISS_DOCSTORE_FILE = "index.pkl"


class VectorStoreError(ValueError):
    """FAISS索引构建、校验或加载失败。"""


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """可审计FAISS索引清单。"""

    schema_version: str
    index_id: str
    created_at: str
    embedding_model: str
    chunk_count: int
    document_count: int
    documents: tuple[dict[str, str], ...]
    file_checksums: dict[str, str]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> IndexManifest:
        """从JSON反序列化并验证必要字段。"""

        required = (
            "schema_version",
            "index_id",
            "created_at",
            "embedding_model",
            "chunk_count",
            "document_count",
            "documents",
            "file_checksums",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise VectorStoreError(f"索引manifest缺少字段：{', '.join(missing)}")
        if payload["schema_version"] != "1.0":
            raise VectorStoreError(
                f"不支持的索引manifest版本：{payload['schema_version']}"
            )
        documents = payload["documents"]
        checksums = payload["file_checksums"]
        if not isinstance(documents, list) or not all(
            isinstance(item, dict) for item in documents
        ):
            raise VectorStoreError("manifest documents必须是object数组")
        if not isinstance(checksums, dict):
            raise VectorStoreError("manifest file_checksums必须是object")
        return cls(
            schema_version=str(payload["schema_version"]),
            index_id=str(payload["index_id"]),
            created_at=str(payload["created_at"]),
            embedding_model=str(payload["embedding_model"]),
            chunk_count=int(payload["chunk_count"]),
            document_count=int(payload["document_count"]),
            documents=tuple(
                {str(key): str(value) for key, value in item.items()}
                for item in documents
            ),
            file_checksums={
                str(key): str(value) for key, value in checksums.items()
            },
        )


class MedicalFAISSStore:
    """LangChain FAISS包装，增加索引版本与完整性验证。"""

    def __init__(
        self,
        store: Any,
        manifest: IndexManifest,
        index_path: Path,
    ) -> None:
        self.store = store
        self.manifest = manifest
        self.index_path = index_path

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        embeddings: Embeddings,
        index_path: str | Path,
        embedding_model: str,
        overwrite: bool = False,
    ) -> MedicalFAISSStore:
        """从文档块构建FAISS索引并保存校验清单。"""

        if not documents:
            raise VectorStoreError("不能使用空文档集合构建FAISS索引")
        if not embedding_model.strip():
            raise VectorStoreError("embedding_model不能为空")

        target = Path(index_path).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        managed_files = (
            target / FAISS_INDEX_FILE,
            target / FAISS_DOCSTORE_FILE,
            target / INDEX_MANIFEST_FILE,
        )
        existing = [path for path in managed_files if path.exists()]
        if existing and not overwrite:
            raise VectorStoreError(
                f"索引文件已存在：{existing[0]}；如需重建请使用overwrite"
            )

        try:
            from langchain_community.vectorstores import FAISS

            store = FAISS.from_documents(list(documents), embeddings)
            store.save_local(str(target))
        except Exception as exc:
            raise VectorStoreError(f"FAISS索引构建失败：{exc}") from exc

        checksums = {
            FAISS_INDEX_FILE: _sha256_file(target / FAISS_INDEX_FILE),
            FAISS_DOCSTORE_FILE: _sha256_file(target / FAISS_DOCSTORE_FILE),
        }
        document_summaries = _document_summaries(documents)
        index_id = _index_id(documents, embedding_model)
        manifest = IndexManifest(
            schema_version="1.0",
            index_id=index_id,
            created_at=datetime.now(UTC).isoformat(),
            embedding_model=embedding_model,
            chunk_count=len(documents),
            document_count=len(document_summaries),
            documents=tuple(document_summaries),
            file_checksums=checksums,
        )
        (target / INDEX_MANIFEST_FILE).write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return cls(store=store, manifest=manifest, index_path=target)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        embeddings: Embeddings,
        expected_embedding_model: str | None = None,
    ) -> MedicalFAISSStore:
        """校验文件哈希后加载可信本地FAISS索引。

        LangChain FAISS docstore使用pickle；只有清单与文件校验值匹配后，才开启
        ``allow_dangerous_deserialization``。仍然只应加载本系统生成的可信索引。
        """

        target = Path(index_path).expanduser().resolve()
        manifest = load_index_manifest(target)
        if (
            expected_embedding_model is not None
            and manifest.embedding_model != expected_embedding_model
        ):
            raise VectorStoreError(
                "索引Embedding模型不匹配："
                f"index={manifest.embedding_model}，runtime={expected_embedding_model}"
            )
        _verify_index_files(target, manifest)

        try:
            from langchain_community.vectorstores import FAISS

            store = FAISS.load_local(
                str(target),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"FAISS索引加载失败：{exc}") from exc
        return cls(store=store, manifest=manifest, index_path=target)

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        metadata_filter: Mapping[str, Any] | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """执行FAISS相似度检索；原始score是L2距离，越小越相关。"""

        if not query.strip():
            raise VectorStoreError("检索问题不能为空")
        if k < 1:
            raise VectorStoreError("k必须大于等于1")
        resolved_fetch_k = fetch_k or max(k * 4, 20)
        try:
            return self.store.similarity_search_with_score(
                query.strip(),
                k=k,
                filter=dict(metadata_filter) if metadata_filter else None,
                fetch_k=resolved_fetch_k,
            )
        except Exception as exc:
            raise VectorStoreError(f"FAISS检索失败：{exc}") from exc


def load_index_manifest(index_path: str | Path) -> IndexManifest:
    """读取索引manifest。"""

    target = Path(index_path).expanduser().resolve()
    manifest_path = target / INDEX_MANIFEST_FILE
    if not manifest_path.is_file():
        raise VectorStoreError(f"索引manifest不存在：{manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorStoreError(f"无法读取索引manifest：{manifest_path}") from exc
    if not isinstance(payload, dict):
        raise VectorStoreError("索引manifest根节点必须是object")
    return IndexManifest.from_mapping(payload)


def _verify_index_files(index_path: Path, manifest: IndexManifest) -> None:
    """校验FAISS和docstore文件未被修改。"""

    for filename in (FAISS_INDEX_FILE, FAISS_DOCSTORE_FILE):
        path = index_path / filename
        expected = manifest.file_checksums.get(filename)
        if not path.is_file() or expected is None:
            raise VectorStoreError(f"索引文件或校验值缺失：{path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise VectorStoreError(f"索引文件完整性校验失败：{filename}")


def _document_summaries(documents: Sequence[Document]) -> list[dict[str, str]]:
    """从chunk元数据汇总索引来源文档。"""

    summaries: dict[str, dict[str, str]] = {}
    for document in documents:
        metadata = document.metadata
        document_id = str(metadata.get("document_id", "")).strip()
        if not document_id:
            raise VectorStoreError("文档块缺少document_id元数据")
        summaries[document_id] = {
            "document_id": document_id,
            "title": str(metadata.get("title", "")),
            "source": str(metadata.get("source", "")),
            "version": str(metadata.get("version", "")),
            "topic": str(metadata.get("topic", "")),
            "file_sha256": str(metadata.get("file_sha256", "")),
        }
    return [summaries[key] for key in sorted(summaries)]


def _index_id(documents: Sequence[Document], embedding_model: str) -> str:
    """根据模型与chunk ID集合生成内容寻址索引ID。"""

    chunk_ids = sorted(
        str(document.metadata.get("chunk_id", "")) for document in documents
    )
    content = "\n".join((embedding_model, *chunk_ids))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def _sha256_file(path: Path) -> str:
    """计算索引文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """创建PDF到FAISS的命令行参数。"""

    parser = argparse.ArgumentParser(description="构建医学影像知识库FAISS索引。")
    parser.add_argument("--pdf-dir", required=True, help="获授权PDF根目录")
    parser.add_argument("--manifest", required=True, help="知识文档manifest.json")
    parser.add_argument("--index-dir", required=True, help="FAISS索引输出目录")
    parser.add_argument("--model", default=None, help="BGE模型，默认BAAI/bge-m3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from rag.loader import ChunkingConfig, KnowledgeLoadError, MedicalPDFLoader

    """加载PDF、生成BGE向量并保存FAISS索引。"""

    args = build_parser().parse_args(argv)
    embedding_config = BGEEmbeddingConfig(
        model_name=args.model or BGEEmbeddingConfig.from_env().model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    try:
        loader = MedicalPDFLoader(
            pdf_root=args.pdf_dir,
            chunking=ChunkingConfig(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            ),
        )
        documents = loader.load_manifest(args.manifest)
        embeddings = create_bge_embeddings(embedding_config)
        store = MedicalFAISSStore.build(
            documents=documents,
            embeddings=embeddings,
            index_path=args.index_dir,
            embedding_model=embedding_config.model_name,
            overwrite=args.overwrite,
        )
    except (KnowledgeLoadError, EmbeddingSetupError, VectorStoreError, OSError) as exc:
        print(f"知识库构建失败：{exc}")
        return 1

    print(f"FAISS索引构建完成：{store.index_path}")
    print(f"索引ID：{store.manifest.index_id}")
    print(f"来源文档：{store.manifest.document_count}")
    print(f"文本块：{store.manifest.chunk_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
