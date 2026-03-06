from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.retrieval.dense_retriever as dense_module
import src.retrieval.reranker as reranker_module
from src.core.constants import RRF_K
from src.core.schemas import EmbeddingRecord, TextChunk
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import FaissStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse_retriever import SparseRetriever


class FakeSentenceTransformer:
    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray([vector], dtype="float32")

    def encode(self, texts, convert_to_numpy=True):  # noqa: ANN001
        return self.vector


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=float)

    def predict(self, pairs):  # noqa: ANN001
        assert len(pairs) == len(self.scores)
        return self.scores


class StubRetriever:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        return [dict(item) for item in self.results[:top_k]]


@pytest.fixture
def retrieval_artifacts(tmp_path: Path) -> dict[str, Path]:
    chunks = [
        TextChunk(
            chunk_id="paper1_aaaabbbb",
            doc_id="paper1",
            text="alpha energy baseline",
            page_number=1,
            headings=["Intro"],
            source_file="paper1.pdf",
        ),
        TextChunk(
            chunk_id="paper2_ccccdddd",
            doc_id="paper2",
            text="beta emissions comparison",
            page_number=2,
            headings=["Results"],
            source_file="paper2.pdf",
        ),
        TextChunk(
            chunk_id="paper3_eeeeffff",
            doc_id="paper3",
            text="gamma water footprint",
            page_number=3,
            headings=["Appendix"],
            source_file="paper3.pdf",
        ),
    ]
    embeddings = [
        EmbeddingRecord(
            id="paper1_aaaabbbb",
            text="alpha energy baseline",
            metadata={
                "doc_id": "paper1",
                "page_number": 1,
                "headings": ["Intro"],
                "source_file": "paper1.pdf",
                "chunk_type": "text",
            },
            embedding=[0.0, 0.0],
            content_hash="hash-paper1",
            embedding_model="mock-model",
        ),
        EmbeddingRecord(
            id="paper2_ccccdddd",
            text="beta emissions comparison",
            metadata={
                "doc_id": "paper2",
                "page_number": 2,
                "headings": ["Results"],
                "source_file": "paper2.pdf",
                "chunk_type": "text",
            },
            embedding=[5.0, 5.0],
            content_hash="hash-paper2",
            embedding_model="mock-model",
        ),
        EmbeddingRecord(
            id="paper3_eeeeffff",
            text="gamma water footprint",
            metadata={
                "doc_id": "paper3",
                "page_number": 3,
                "headings": ["Appendix"],
                "source_file": "paper3.pdf",
                "chunk_type": "text",
            },
            embedding=[10.0, 10.0],
            content_hash="hash-paper3",
            embedding_model="mock-model",
        ),
    ]

    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(chunk.model_dump(mode="json")) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    embeddings_path.write_text(
        "".join(json.dumps(record.model_dump(mode="json")) + "\n" for record in embeddings),
        encoding="utf-8",
    )

    return {"chunks_path": chunks_path, "embeddings_path": embeddings_path}


def test_dense_retrieval_returns_results(monkeypatch, tmp_path: Path, retrieval_artifacts: dict[str, Path]):
    monkeypatch.setattr(dense_module, "_embedding_model", lambda model_name: FakeSentenceTransformer([0.0, 0.0]))
    store = FaissStore(
        index_path=tmp_path / "faiss.index",
        text_data_path=tmp_path / "text_data.pkl",
        embeddings_path=retrieval_artifacts["embeddings_path"],
    )

    results = DenseRetriever(store=store, model_name="mock-model", distance_threshold=100.0).retrieve(
        "alpha energy",
        top_k=2,
    )

    assert [item["ref_id"] for item in results] == ["paper1", "paper2"]
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["page"] == 1
    assert results[0]["source_file"] == "paper1.pdf"


def test_dense_retrieval_respects_threshold(monkeypatch, tmp_path: Path, retrieval_artifacts: dict[str, Path]):
    monkeypatch.setattr(
        dense_module,
        "_embedding_model",
        lambda model_name: FakeSentenceTransformer([100.0, 100.0]),
    )
    store = FaissStore(
        index_path=tmp_path / "faiss.index",
        text_data_path=tmp_path / "text_data.pkl",
        embeddings_path=retrieval_artifacts["embeddings_path"],
    )

    results = DenseRetriever(store=store, model_name="mock-model", distance_threshold=1.0).retrieve(
        "no match",
        top_k=3,
    )

    assert results == []


def test_sparse_retrieval_returns_results(tmp_path: Path, retrieval_artifacts: dict[str, Path]):
    index = BM25Index(index_path=tmp_path / "bm25_index.pkl", chunks_path=retrieval_artifacts["chunks_path"])

    results = SparseRetriever(index=index).retrieve("emissions comparison", top_k=2)

    assert len(results) == 1
    assert results[0]["ref_id"] == "paper2"
    assert results[0]["chunk_id"] == "paper2_ccccdddd"
    assert results[0]["score"] > 0.0


def test_hybrid_scores_are_rrf():
    dense_results = [
        {
            "rank": 1,
            "chunk_id": "paper1_aaaabbbb",
            "ref_id": "paper1",
            "score": 0.9,
            "text": "alpha",
            "page": 1,
            "source_file": "paper1.pdf",
            "headings": [],
            "chunk_type": "text",
        },
        {
            "rank": 2,
            "chunk_id": "paper2_ccccdddd",
            "ref_id": "paper2",
            "score": 0.8,
            "text": "beta",
            "page": 2,
            "source_file": "paper2.pdf",
            "headings": [],
            "chunk_type": "text",
        },
    ]
    sparse_results = [
        {
            "rank": 1,
            "chunk_id": "paper1_aaaabbbb",
            "ref_id": "paper1",
            "score": 0.7,
            "text": "alpha",
            "page": 1,
            "source_file": "paper1.pdf",
            "headings": [],
            "chunk_type": "text",
        },
        {
            "rank": 2,
            "chunk_id": "paper3_eeeeffff",
            "ref_id": "paper3",
            "score": 0.6,
            "text": "gamma",
            "page": 3,
            "source_file": "paper3.pdf",
            "headings": [],
            "chunk_type": "text",
        },
    ]

    results = HybridRetriever(
        dense_retriever=StubRetriever(dense_results),
        sparse_retriever=StubRetriever(sparse_results),
    ).retrieve("question", top_k=3)

    assert results[0]["chunk_id"] == "paper1_aaaabbbb"
    assert results[0]["score"] == pytest.approx((1 / (RRF_K + 1)) + (1 / (RRF_K + 1)))
    remaining = {item["chunk_id"]: item["score"] for item in results[1:]}
    assert remaining["paper2_ccccdddd"] == pytest.approx(1 / (RRF_K + 2))
    assert remaining["paper3_eeeeffff"] == pytest.approx(1 / (RRF_K + 2))
    assert [item["rank"] for item in results] == [1, 2, 3]


def test_reranker_changes_order(monkeypatch):
    monkeypatch.setattr(reranker_module, "_cross_encoder", lambda model_name: FakeCrossEncoder([0.1, 0.9]))
    candidates = [
        {
            "rank": 1,
            "chunk_id": "paper1_aaaabbbb",
            "ref_id": "paper1",
            "score": 0.8,
            "text": "alpha",
        },
        {
            "rank": 2,
            "chunk_id": "paper2_ccccdddd",
            "ref_id": "paper2",
            "score": 0.7,
            "text": "beta",
        },
    ]

    results = Reranker(model_name="mock-reranker").rerank("question", candidates, top_k=2)

    assert [item["chunk_id"] for item in results] == ["paper2_ccccdddd", "paper1_aaaabbbb"]
    assert [item["rank"] for item in results] == [1, 2]
