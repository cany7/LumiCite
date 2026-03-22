from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator

TEXT_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+_[0-9a-f]{8}$")
FIGURE_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+_fig_[0-9a-f]{8}$")
TABLE_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+_tab_[0-9a-f]{8}$")


def _validate_iso8601(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("must be a valid ISO 8601 timestamp") from exc
    return value


def _normalize_string(value: str) -> str:
    return value.strip()


def _normalize_string_list(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if item:
            normalized.append(item)
    return normalized


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class BaseChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: ChunkType
    page_number: int | None = None
    headings: list[str] = Field(default_factory=list)

    @field_validator("chunk_id", "doc_id")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        normalized = _normalize_string(value)
        if not normalized:
            raise ValueError("identifier must be non-empty")
        return normalized

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _normalize_string(value)

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("page_number must be positive when provided")
        return value

    @field_validator("headings")
    @classmethod
    def validate_headings(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)

    @model_validator(mode="after")
    def validate_chunk_id(self) -> "BaseChunk":
        if not self.chunk_id.startswith(f"{self.doc_id}_"):
            raise ValueError("chunk_id must start with '{doc_id}_'")

        if self.chunk_type == ChunkType.TEXT and not TEXT_CHUNK_ID_RE.match(self.chunk_id):
            raise ValueError("text chunk_id must match '{doc_id}_{8hex}'")
        if self.chunk_type == ChunkType.FIGURE and not FIGURE_CHUNK_ID_RE.match(self.chunk_id):
            raise ValueError("figure chunk_id must match '{doc_id}_fig_{8hex}'")
        if self.chunk_type == ChunkType.TABLE and not TABLE_CHUNK_ID_RE.match(self.chunk_id):
            raise ValueError("table chunk_id must match '{doc_id}_tab_{8hex}'")
        return self


class TextChunk(BaseChunk):
    chunk_type: ChunkType = ChunkType.TEXT

    @field_validator("text")
    @classmethod
    def validate_text_non_empty(cls, value: str) -> str:
        normalized = _normalize_string(value)
        if not normalized:
            raise ValueError("text must be non-empty and not whitespace-only")
        return normalized


class TableChunk(BaseChunk):
    chunk_type: ChunkType = ChunkType.TABLE
    caption: str = ""
    footnotes: list[str] = Field(default_factory=list)
    asset_path: str = ""

    @field_validator("text")
    @classmethod
    def validate_text_non_empty(cls, value: str) -> str:
        normalized = _normalize_string(value)
        if not normalized:
            raise ValueError("text must be non-empty and not whitespace-only")
        return normalized

    @field_validator("caption", "asset_path")
    @classmethod
    def validate_table_strings(cls, value: str) -> str:
        return _normalize_string(value)

    @field_validator("footnotes")
    @classmethod
    def validate_footnotes(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)


class FigureChunk(BaseChunk):
    chunk_type: ChunkType = ChunkType.FIGURE
    caption: str = ""
    footnotes: list[str] = Field(default_factory=list)
    asset_path: str = ""

    @field_validator("caption", "asset_path")
    @classmethod
    def validate_figure_strings(cls, value: str) -> str:
        return _normalize_string(value)

    @field_validator("footnotes")
    @classmethod
    def validate_footnotes(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)


ChunkModel: TypeAlias = TextChunk | TableChunk | FigureChunk


class EmbeddingRecord(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]
    content_hash: str
    embedding_model: str
    created_at: str = ""

    @field_validator("id", "content_hash", "embedding_model")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        normalized = _normalize_string(value)
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _normalize_string(value)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        if not value:
            return value
        return _validate_iso8601(value)


class ManifestEntry(BaseModel):
    doc_id: str
    content_hash: str
    file_size_bytes: int
    parsed_at: str
    num_chunks: int
    embedding_model: str
    embedded_at: str
    status: str
    error_message: str = ""

    @field_validator("doc_id", "content_hash", "embedding_model", "status", "error_message")
    @classmethod
    def validate_manifest_strings(cls, value: str) -> str:
        return _normalize_string(value)

    @field_validator("parsed_at", "embedded_at")
    @classmethod
    def validate_manifest_timestamps(cls, value: str) -> str:
        return _validate_iso8601(value)


class SearchResult(BaseModel):
    rank: int
    doc_id: str
    chunk_id: str
    chunk_type: ChunkType
    score: float
    text: str
    page_number: int | None = None
    headings: list[str] = Field(default_factory=list)
    caption: str = ""
    asset_path: str = ""

    @field_validator("doc_id", "chunk_id", "text")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        normalized = _normalize_string(value)
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("caption", "asset_path")
    @classmethod
    def validate_optional_strings(cls, value: str) -> str:
        return _normalize_string(value)

    @field_validator("headings")
    @classmethod
    def validate_headings(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    page_number: int | None = None
    evidence_text: str
    evidence_type: ChunkType
    headings: list[str] = Field(default_factory=list)
    caption: str = ""
    asset_path: str = ""

    @field_validator("doc_id", "chunk_id", "evidence_text")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        normalized = _normalize_string(value)
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("caption", "asset_path")
    @classmethod
    def validate_optional_strings(cls, value: str) -> str:
        return _normalize_string(value)

    @field_validator("headings")
    @classmethod
    def validate_headings(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)


class VerificationResult(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    corrected_output: dict[str, Any] | None = None

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, value: list[str]) -> list[str]:
        return _normalize_string_list(value)


class RAGAnswer(BaseModel):
    answer: str
    supporting_materials: str
    explanation: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    retrieval_mode: str | None = None
    llm_backend: str | None = None
    verification: VerificationResult | None = None

    @field_validator(
        "answer",
        "supporting_materials",
        "explanation",
        "retrieval_mode",
        "llm_backend",
        mode="before",
    )
    @classmethod
    def validate_optional_strings(cls, value: Any) -> Any:
        if value is None:
            return value
        return _normalize_string(str(value))


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
    per_question: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _validate_iso8601(value)
