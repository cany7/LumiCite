from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+(?:_(?:img|fig|tab))?_[0-9a-f]{8}$")


def _validate_iso8601(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("must be a valid ISO 8601 timestamp") from exc
    return value


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: ChunkType = ChunkType.TEXT
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    source_file: str = ""

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("text must be non-empty and not whitespace-only")
        return text

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not CHUNK_ID_RE.match(value):
            raise ValueError(
                "chunk_id must match '{doc_id}_{8hex}' or legacy '{doc_id}_{img|fig|tab}_{8hex}'"
            )
        return value


class TableChunk(TextChunk):
    chunk_type: ChunkType = ChunkType.TABLE
    column_headers: list[str] = Field(default_factory=list)
    row_headers: list[str] = Field(default_factory=list)
    key_values: dict[str, str] = Field(default_factory=dict)
    caption: str = ""


class FigureChunk(TextChunk):
    chunk_type: ChunkType = ChunkType.FIGURE
    chart_type: str = ""
    axis_labels: dict[str, str] = Field(default_factory=dict)
    trends: list[str] = Field(default_factory=list)
    caption: str = ""


class EmbeddingRecord(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]
    content_hash: str
    embedding_model: str
    created_at: str = ""

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("text must be non-empty and not whitespace-only")
        return text

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        if not value:
            return value
        return _validate_iso8601(value)


class ManifestEntry(BaseModel):
    content_hash: str
    file_size_bytes: int
    parsed_at: str
    parser_version: str
    chunk_strategy: str
    num_chunks: int
    embedding_model: str
    embedded_at: str
    status: str
    error_message: str = ""

    @field_validator("parsed_at", "embedded_at")
    @classmethod
    def validate_manifest_timestamps(cls, value: str) -> str:
        return _validate_iso8601(value)


class Citation(BaseModel):
    ref_id: str
    page: int | None = None
    evidence_text: str
    evidence_type: str


class VerificationResult(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str]
    corrected_output: dict | None = None


class RAGAnswer(BaseModel):
    answer: str
    answer_value: str
    answer_unit: str
    ref_id: list[str]
    supporting_materials: str
    explanation: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    retrieval_mode: str | None = None
    llm_backend: str | None = None
    verification: VerificationResult | None = None


class BenchmarkReport(BaseModel):
    run_id: str
    tag: str
    timestamp: str
    config_hash: str
    git_commit: str
    dataset: str
    retrieval_mode: str
    top_k: int
    reranker_enabled: bool
    embedding_model: str
    num_questions: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_retrieval_latency_ms: float
    p95_retrieval_latency_ms: float
    per_question: list[dict] = Field(default_factory=list)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _validate_iso8601(value)
