"""非PDF文本知识的兼容入库入口。"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embedding import BGEEmbeddingConfig, create_bge_embeddings
from rag.schemas import KnowledgeDocument
from rag.vector_store import MedicalFAISSStore


class KnowledgeIngestionPipeline:
    """Parse, chunk, embed, and index approved medical documents."""

    def __init__(
        self,
        index_path: Path,
        embeddings: Embeddings | None = None,
        embedding_config: BGEEmbeddingConfig | None = None,
    ) -> None:
        self.index_path = index_path
        self.embedding_config = embedding_config or BGEEmbeddingConfig.from_env()
        self.embeddings = embeddings

    def ingest(
        self,
        documents: Sequence[KnowledgeDocument],
        overwrite: bool = False,
    ) -> str:
        """Create a new immutable knowledge index version."""

        if not documents:
            raise ValueError("documents不能为空")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            add_start_index=True,
        )
        langchain_documents = [
            Document(
                page_content=document.content,
                metadata={
                    **document.metadata,
                    "document_id": document.document_id,
                    "title": document.title,
                    "source": document.source,
                    "version": document.version,
                    "topic": document.metadata.get("topic", "glioma_mri"),
                    "citation": f"{document.title} ({document.version})",
                },
            )
            for document in documents
        ]
        chunks = splitter.split_documents(langchain_documents)
        for chunk in chunks:
            identity = (
                f"{chunk.metadata['document_id']}|"
                f"{chunk.metadata.get('start_index', 0)}|{chunk.page_content}"
            )
            chunk.metadata["chunk_id"] = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:24]
            chunk.metadata.setdefault("file_sha256", "")

        embeddings = self.embeddings or create_bge_embeddings(self.embedding_config)
        store = MedicalFAISSStore.build(
            documents=chunks,
            embeddings=embeddings,
            index_path=self.index_path,
            embedding_model=self.embedding_config.model_name,
            overwrite=overwrite,
        )
        return store.manifest.index_id
