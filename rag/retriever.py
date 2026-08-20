"""带来源引用的医学影像知识FAISS Retriever。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from rag.embedding import (
    BGEEmbeddingConfig,
    EmbeddingSetupError,
    create_bge_embeddings,
)
from rag.schemas import ALLOWED_TOPICS, RetrievedChunk
from rag.vector_store import MedicalFAISSStore, VectorStoreError, load_index_manifest


class RetrievalError(ValueError):
    """医学知识检索请求或结果不合法。"""


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """医生问题、证据块和可直接传给LLM的引用上下文。"""

    query: str
    chunks: tuple[RetrievedChunk, ...]

    def format_context(self) -> str:
        """将证据格式化为带编号和引用的RAG上下文。"""

        if not self.chunks:
            return "未检索到满足条件的医学资料。"
        sections = []
        for index, chunk in enumerate(self.chunks, start=1):
            sections.append(
                f"[资料{index}] {chunk.citation}\n"
                f"相关度：{chunk.score:.4f}\n"
                f"{chunk.text}"
            )
        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """转换为API/Agent可序列化结构。"""

        return {
            "query": self.query,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "context": self.format_context(),
            "requires_source_verification": True,
        }


class MedicalKnowledgeRetriever:
    """从已发布FAISS索引检索可追溯医学证据。"""

    def __init__(
        self,
        index_path: str | Path,
        embeddings: Embeddings | None = None,
        embedding_config: BGEEmbeddingConfig | None = None,
        vector_store: MedicalFAISSStore | None = None,
    ) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        self.embedding_config = embedding_config
        self._embeddings = embeddings
        self._vector_store = vector_store

    def _get_store(self) -> MedicalFAISSStore:
        """首次检索时加载模型和索引，避免服务启动阶段阻塞。"""

        if self._vector_store is not None:
            return self._vector_store
        manifest = load_index_manifest(self.index_path)
        config = self.embedding_config or BGEEmbeddingConfig(
            model_name=manifest.embedding_model
        )
        if config.model_name != manifest.embedding_model:
            raise RetrievalError(
                "Retriever模型与索引不匹配："
                f"index={manifest.embedding_model}，runtime={config.model_name}"
            )
        embeddings = self._embeddings or create_bge_embeddings(config)
        self._vector_store = MedicalFAISSStore.load(
            index_path=self.index_path,
            embeddings=embeddings,
            expected_embedding_model=config.model_name,
        )
        return self._vector_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        topic: str | None = None,
        score_threshold: float = 0.0,
    ) -> tuple[RetrievedChunk, ...]:
        """返回按相关度排序的证据块，score范围0至1且越大越相关。"""

        normalized_query = query.strip()
        if not normalized_query:
            raise RetrievalError("医生问题不能为空")
        if top_k < 1 or top_k > 50:
            raise RetrievalError("top_k必须位于1至50之间")
        if not 0.0 <= score_threshold <= 1.0:
            raise RetrievalError("score_threshold必须位于0至1之间")
        if topic is not None and topic not in ALLOWED_TOPICS:
            raise RetrievalError(
                f"topic必须是以下之一：{', '.join(sorted(ALLOWED_TOPICS))}"
            )

        metadata_filter: Mapping[str, str] | None = (
            {"topic": topic} if topic else None
        )
        results = self._get_store().similarity_search_with_score(
            query=normalized_query,
            k=top_k,
            metadata_filter=metadata_filter,
        )

        chunks: list[RetrievedChunk] = []
        for document, distance in results:
            relevance = _distance_to_relevance(float(distance))
            if relevance < score_threshold:
                continue
            metadata = dict(document.metadata)
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    document_id=str(metadata.get("document_id", "")),
                    text=document.page_content,
                    score=round(relevance, 6),
                    citation=_citation(metadata),
                    metadata=metadata,
                )
            )
        return tuple(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        topic: str | None = None,
        score_threshold: float = 0.0,
    ) -> RetrievalResponse:
        """返回证据块和拼接后的LLM上下文。"""

        return RetrievalResponse(
            query=query.strip(),
            chunks=self.search(
                query=query,
                top_k=top_k,
                topic=topic,
                score_threshold=score_threshold,
            ),
        )


def _distance_to_relevance(distance: float) -> float:
    """将非负L2距离单调映射为0至1相关度。"""

    if distance < 0:
        distance = 0.0
    return 1.0 / (1.0 + distance)


def _citation(metadata: Mapping[str, Any]) -> str:
    """优先使用入库时生成的引用，否则从元数据重建。"""

    citation = str(metadata.get("citation", "")).strip()
    if citation:
        return citation
    title = str(metadata.get("title", "Untitled"))
    version = str(metadata.get("version", "unknown version"))
    page = str(metadata.get("page", "?"))
    return f"{title} ({version}), p.{page}"


def build_parser() -> argparse.ArgumentParser:
    """创建医学知识检索命令行参数。"""

    parser = argparse.ArgumentParser(description="检索本地医学影像FAISS知识库。")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--topic",
        choices=[
            "who_cns",
            "nccn_glioma",
            "glioma_mri",
            "follow_up_criteria",
        ],
        default=None,
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="输出标准JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行一次医生问题检索。"""

    args = build_parser().parse_args(argv)
    try:
        response = MedicalKnowledgeRetriever(args.index_dir).retrieve(
            query=args.query,
            top_k=args.top_k,
            topic=args.topic,
            score_threshold=args.score_threshold,
        )
    except (
        RetrievalError,
        VectorStoreError,
        EmbeddingSetupError,
        OSError,
    ) as exc:
        print(f"医学知识检索失败：{exc}")
        return 1

    if args.json:
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(response.format_context())
        print("\n提示：检索资料须核对原文版本和页码，不作为独立临床决策依据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
