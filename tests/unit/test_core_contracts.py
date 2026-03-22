from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.schemas import (
    BenchmarkReport,
    ChunkType,
    Citation,
    EmbeddingRecord,
    FigureChunk,
    ManifestEntry,
    RAGAnswer,
    SearchResult,
    TableChunk,
    TextChunk,
    VerificationResult,
)


def test_chunk_models_enforce_id_formats_defaults_and_visual_fields() -> None:
    text_chunk = TextChunk(chunk_id="paper1_deadbeef", doc_id="paper1", text="  alpha  ")
    figure_chunk = FigureChunk(
        chunk_id="paper1_fig_cafebabe",
        doc_id="paper1",
        text="summary",
        page_number=2,
        headings=[" Results ", " "],
        caption=" Figure 1. Trend ",
        footnotes=[" n=1 ", "", " p<0.05 "],
        asset_path=" data/assets/paper1/figure.png ",
    )
    table_chunk = TableChunk(
        chunk_id="paper1_tab_1234abcd",
        doc_id="paper1",
        text="table body",
        caption=" Table 1. Scores ",
        footnotes=[" macro avg "],
        asset_path=" data/assets/paper1/table.png ",
    )

    assert text_chunk.text == "alpha"
    assert text_chunk.headings == []
    assert not hasattr(text_chunk, "caption")
    assert figure_chunk.headings == ["Results"]
    assert figure_chunk.caption == "Figure 1. Trend"
    assert figure_chunk.footnotes == ["n=1", "p<0.05"]
    assert figure_chunk.asset_path == "data/assets/paper1/figure.png"
    assert table_chunk.caption == "Table 1. Scores"
    assert table_chunk.footnotes == ["macro avg"]

    with pytest.raises(ValidationError, match="chunk_id must start with"):
        TextChunk(chunk_id="paper2_deadbeef", doc_id="paper1", text="alpha")

    with pytest.raises(ValidationError, match="figure chunk_id must match"):
        FigureChunk(chunk_id="paper1_deadbeef", doc_id="paper1", text="summary")

    with pytest.raises(ValidationError, match="page_number must be positive"):
        TextChunk(chunk_id="paper1_deadbeef", doc_id="paper1", text="alpha", page_number=0)


def test_search_and_citation_models_support_visual_fields() -> None:
    result = SearchResult(
        rank=1,
        doc_id="paper1",
        chunk_id="paper1_fig_deadbeef",
        chunk_type=ChunkType.FIGURE,
        score=0.9,
        text="figure summary",
        page_number=5,
        headings=["3 Results"],
        caption="Figure 3. Accuracy",
        asset_path="data/assets/paper1/paper1_fig_deadbeef.png",
    )
    citation = Citation(
        doc_id="paper1",
        chunk_id="paper1_fig_deadbeef",
        page_number=5,
        evidence_text="figure summary",
        evidence_type=ChunkType.FIGURE,
        headings=["3 Results"],
        caption="Figure 3. Accuracy",
        asset_path="data/assets/paper1/paper1_fig_deadbeef.png",
    )

    assert result.chunk_type is ChunkType.FIGURE
    assert result.caption == "Figure 3. Accuracy"
    assert citation.evidence_type is ChunkType.FIGURE
    assert citation.asset_path.endswith(".png")


def test_embedding_manifest_and_benchmark_models_validate_timestamps() -> None:
    record = EmbeddingRecord(
        id="paper1_deadbeef",
        text="alpha",
        metadata={
            "doc_id": "paper1",
            "page_number": 1,
            "headings": ["Intro"],
            "chunk_type": "text",
            "caption": "",
            "asset_path": "",
        },
        embedding=[0.1, 0.2],
        content_hash="hash",
        embedding_model="mock-model",
        created_at="2026-03-21T10:00:00Z",
    )
    manifest = ManifestEntry(
        doc_id="paper1",
        content_hash="hash",
        file_size_bytes=123,
        parsed_at="2026-03-21T10:00:00Z",
        num_chunks=2,
        embedding_model="mock-model",
        embedded_at="2026-03-21T10:05:00Z",
        status="complete",
    )
    report = BenchmarkReport(
        run_id="run-1",
        tag="phase4",
        timestamp="2026-03-21T10:00:00Z",
        config_hash="cfg",
        git_commit="abc123",
        dataset="data/train.csv",
        retrieval_mode="hybrid",
        top_k=5,
        reranker_enabled=True,
        embedding_model="mock-model",
        num_questions=2,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        mean_retrieval_latency_ms=10.0,
        p95_retrieval_latency_ms=15.0,
    )

    assert record.created_at == "2026-03-21T10:00:00Z"
    assert manifest.status == "complete"
    assert report.timestamp == "2026-03-21T10:00:00Z"

    with pytest.raises(ValidationError, match="ISO 8601"):
        ManifestEntry(
            doc_id="paper1",
            content_hash="hash",
            file_size_bytes=123,
            parsed_at="not-a-time",
            num_chunks=2,
            embedding_model="mock-model",
            embedded_at="2026-03-21T10:05:00Z",
            status="complete",
        )


def test_rag_answer_and_verification_optional_fields_behave_as_contract() -> None:
    verification = VerificationResult(passed=True, confidence=0.9, warnings=[" warning ", ""], corrected_output=None)
    answer = RAGAnswer(
        answer="552 tCO2e",
        supporting_materials=" evidence ",
        explanation=" because ",
        llm_backend=" api ",
        retrieval_mode=" hybrid ",
        verification=verification,
    )

    assert answer.citations == []
    assert answer.llm_backend == "api"
    assert answer.retrieval_mode == "hybrid"
    assert answer.verification is not None
    assert answer.verification.warnings == ["warning"]
