from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from src.core.schemas import (
    BenchmarkReport,
    ChunkType,
    EmbeddingRecord,
    FigureChunk,
    ManifestEntry,
    TableChunk,
    TextChunk,
    VerificationResult,
)


def test_text_chunk_valid_and_strips_text():
    chunk = TextChunk(chunk_id="p1_abc12345", doc_id="p1", text="  hello  ")

    assert chunk.text == "hello"
    assert chunk.chunk_type is ChunkType.TEXT


def test_text_chunk_empty_text_rejected():
    with pytest.raises(ValidationError):
        TextChunk(chunk_id="p1_abc12345", doc_id="p1", text="   ")


def test_table_chunk_inherits_text_fields():
    chunk = TableChunk(
        chunk_id="p1_tab_deadbeef",
        doc_id="p1",
        text="table text",
        column_headers=["A"],
        key_values={"GPU energy": "1287 kWh"},
    )

    assert chunk.chunk_type is ChunkType.TABLE
    assert chunk.column_headers == ["A"]
    assert chunk.key_values == {"GPU energy": "1287 kWh"}
    assert chunk.doc_id == "p1"


def test_figure_chunk_defaults():
    chunk = FigureChunk(chunk_id="p1_fig_deadbeef", doc_id="p1", text="figure text")

    assert chunk.chunk_type is ChunkType.FIGURE
    assert chunk.axis_labels == {}
    assert chunk.trends == []


def test_chunk_id_format_rejected_when_invalid():
    with pytest.raises(ValidationError):
        TextChunk(chunk_id="bad_chunk_id", doc_id="p1", text="hello")


def test_embedding_record_valid_and_hash_deterministic():
    text = "embedded text"
    expected_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

    record = EmbeddingRecord(
        id="p1_abc12345",
        text=text,
        metadata={"doc_id": "p1", "chunk_type": "text"},
        embedding=[0.1, 0.2, 0.3],
        content_hash=expected_hash,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        created_at="2026-03-05T12:34:56Z",
    )

    assert record.content_hash == expected_hash
    assert record.embedding == [0.1, 0.2, 0.3]


def test_embedding_record_empty_text_rejected():
    with pytest.raises(ValidationError):
        EmbeddingRecord(
            id="p1_abc12345",
            text="\n\t",
            metadata={},
            embedding=[0.1],
            content_hash="x",
            embedding_model="model",
        )


def test_embedding_record_invalid_iso_timestamp_rejected():
    with pytest.raises(ValidationError):
        EmbeddingRecord(
            id="p1_abc12345",
            text="hello",
            metadata={},
            embedding=[0.1],
            content_hash="x",
            embedding_model="model",
            created_at="not-iso",
        )


def test_verification_confidence_must_be_in_range():
    with pytest.raises(ValidationError):
        VerificationResult(passed=True, confidence=1.5, warnings=[])


def test_manifest_entry_requires_iso_timestamps():
    with pytest.raises(ValidationError):
        ManifestEntry(
            content_hash="abc",
            file_size_bytes=1,
            parsed_at="not-iso",
            parser_version="docling-1",
            chunk_strategy="hybrid_512_50",
            num_chunks=1,
            embedding_model="model",
            embedded_at="2026-03-05T12:34:56Z",
            status="complete",
        )


def test_benchmark_report_requires_iso_timestamp():
    with pytest.raises(ValidationError):
        BenchmarkReport(
            run_id="run_1",
            tag="run",
            timestamp="not-iso",
            config_hash="abc",
            git_commit="",
            dataset="data.csv",
            retrieval_mode="hybrid",
            top_k=5,
            reranker_enabled=True,
            embedding_model="model",
            num_questions=1,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            mean_retrieval_latency_ms=1.0,
            p95_retrieval_latency_ms=1.0,
        )


def test_chunk_type_enum_round_trips():
    assert ChunkType("text") is ChunkType.TEXT
