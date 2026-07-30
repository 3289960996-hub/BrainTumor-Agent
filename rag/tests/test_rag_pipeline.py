"""医学PDF切分、Embedding配置、FAISS和Retriever测试。"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.embedding import BGEEmbeddingConfig, create_bge_embeddings
from rag.loader import ChunkingConfig, KnowledgeLoadError, MedicalPDFLoader, load_manifest
from rag.retriever import MedicalKnowledgeRetriever
from rag.vector_store import (
    FAISS_DOCSTORE_FILE,
    MedicalFAISSStore,
    VectorStoreError,
)


class _FakePDFLoader:
    def load(self) -> list[Document]:
        return [
            Document(
                page_content=(
                    "增强区域反映对比剂相关信号变化。\n\n"
                    "在胶质瘤影像评估中，应结合增强范围、非增强成分和既往检查。"
                )
                * 4,
                metadata={"page": 0},
            ),
            Document(
                page_content=("随访时需要比较可测量病灶及临床状态。" * 10),
                metadata={"page": 1},
            ),
        ]


class _KeywordEmbeddings(Embeddings):
    """无需模型下载的确定性测试Embedding。"""

    keywords = ("增强", "随访", "分类", "水肿")

    def _embed(self, text: str) -> list[float]:
        vector = np.asarray(
            [float(text.count(keyword)) for keyword in self.keywords] + [0.1],
            dtype=np.float32,
        )
        vector /= np.linalg.norm(vector)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "file": "glioma_mri.pdf",
                        "document_id": "glioma-mri-2026",
                        "title": "Glioma MRI Imaging Review",
                        "source": "Institutional licensed reference",
                        "version": "2026.1",
                        "topic": "glioma_mri",
                        "publication_date": "2026-01-01",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_and_pdf_chunks_keep_page_citations(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    (pdf_root / "glioma_mri.pdf").write_bytes(b"fake-pdf-for-injected-loader")
    manifest_path = _manifest(tmp_path / "manifest.json")

    specs = load_manifest(manifest_path)
    loader = MedicalPDFLoader(
        pdf_root=pdf_root,
        chunking=ChunkingConfig(
            chunk_size=240,
            chunk_overlap=40,
            minimum_chunk_length=20,
        ),
        pdf_loader_factory=lambda _: _FakePDFLoader(),
    )
    chunks = loader.load_manifest(manifest_path)

    assert specs[0].topic == "glioma_mri"
    assert chunks
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2}
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)
    assert all("p." in chunk.metadata["citation"] for chunk in chunks)
    assert all(len(chunk.metadata["file_sha256"]) == 64 for chunk in chunks)


def test_manifest_rejects_path_traversal_at_load_time(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    spec = {
        "documents": [
            {
                "file": "../outside.pdf",
                "document_id": "outside-doc",
                "title": "Outside",
                "source": "test",
                "version": "1",
                "topic": "glioma_mri",
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(spec), encoding="utf-8")
    loader = MedicalPDFLoader(
        pdf_root,
        pdf_loader_factory=lambda _: _FakePDFLoader(),
    )

    with pytest.raises(KnowledgeLoadError, match="越出"):
        loader.load_manifest(manifest_path)


def test_embedding_factory_receives_bge_m3_normalization() -> None:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> Embeddings:
        captured.update(kwargs)
        return _KeywordEmbeddings()

    embeddings = create_bge_embeddings(
        BGEEmbeddingConfig(device="cpu", batch_size=8),
        factory=factory,
    )

    assert isinstance(embeddings, _KeywordEmbeddings)
    assert captured["model_name"] == "BAAI/bge-m3"
    assert captured["model_kwargs"]["device"] == "cpu"
    assert captured["encode_kwargs"]["normalize_embeddings"] is True
    assert captured["encode_kwargs"]["batch_size"] == 8


def _indexed_documents() -> list[Document]:
    return [
        Document(
            page_content="增强区域在胶质瘤评估中用于描述对比增强成分及其变化。",
            metadata={
                "chunk_id": "chunk-enhancing",
                "document_id": "glioma-mri",
                "title": "Glioma MRI Review",
                "source": "licensed source",
                "version": "2026",
                "topic": "glioma_mri",
                "page": 12,
                "citation": "Glioma MRI Review (2026), p.12",
                "file_sha256": "a" * 64,
            },
        ),
        Document(
            page_content="随访评价需比较病灶变化并结合临床状态。",
            metadata={
                "chunk_id": "chunk-follow-up",
                "document_id": "follow-up",
                "title": "Follow-up Criteria",
                "source": "licensed source",
                "version": "2025",
                "topic": "follow_up_criteria",
                "page": 8,
                "citation": "Follow-up Criteria (2025), p.8",
                "file_sha256": "b" * 64,
            },
        ),
    ]


def test_faiss_build_load_and_retrieve_traceable_context(tmp_path: Path) -> None:
    embeddings = _KeywordEmbeddings()
    store = MedicalFAISSStore.build(
        documents=_indexed_documents(),
        embeddings=embeddings,
        index_path=tmp_path / "index",
        embedding_model="test-keyword-embedding",
    )

    assert store.manifest.chunk_count == 2
    assert store.manifest.document_count == 2

    retriever = MedicalKnowledgeRetriever(
        index_path=tmp_path / "index",
        embeddings=embeddings,
        embedding_config=BGEEmbeddingConfig(
            model_name="test-keyword-embedding",
            device="cpu",
        ),
    )
    response = retriever.retrieve("增强区域在胶质瘤评估中的意义？", top_k=1)

    assert len(response.chunks) == 1
    assert response.chunks[0].chunk_id == "chunk-enhancing"
    assert response.chunks[0].score > 0.9
    assert "Glioma MRI Review (2026), p.12" in response.format_context()
    assert response.to_dict()["requires_source_verification"] is True


def test_faiss_topic_filter(tmp_path: Path) -> None:
    embeddings = _KeywordEmbeddings()
    store = MedicalFAISSStore.build(
        documents=_indexed_documents(),
        embeddings=embeddings,
        index_path=tmp_path / "index",
        embedding_model="test-keyword-embedding",
    )
    retriever = MedicalKnowledgeRetriever(
        index_path=tmp_path / "index",
        vector_store=store,
    )

    chunks = retriever.search(
        "随访评价标准",
        top_k=2,
        topic="follow_up_criteria",
    )

    assert len(chunks) == 1
    assert chunks[0].document_id == "follow-up"


def test_tampered_faiss_docstore_is_rejected(tmp_path: Path) -> None:
    embeddings = _KeywordEmbeddings()
    index_path = tmp_path / "index"
    MedicalFAISSStore.build(
        documents=_indexed_documents(),
        embeddings=embeddings,
        index_path=index_path,
        embedding_model="test-keyword-embedding",
    )
    docstore_path = index_path / FAISS_DOCSTORE_FILE
    docstore_path.write_bytes(docstore_path.read_bytes() + b"tampered")

    with pytest.raises(VectorStoreError, match="完整性"):
        MedicalFAISSStore.load(
            index_path=index_path,
            embeddings=embeddings,
            expected_embedding_model="test-keyword-embedding",
        )
