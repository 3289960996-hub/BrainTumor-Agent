"""Knowledge-base document and retrieval result contracts."""

from dataclasses import dataclass, field
from typing import Any

ALLOWED_TOPICS = {
    "who_cns",
    "nccn_glioma",
    "glioma_mri",
    "follow_up_criteria",
}


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """Versioned source document approved for ingestion."""

    document_id: str
    title: str
    source: str
    version: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One traceable evidence chunk returned by retrieval."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    citation: str
    metadata: dict[str, Any] = field(default_factory=dict)
