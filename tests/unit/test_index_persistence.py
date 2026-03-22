from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import src.indexing.bm25_index as bm25_module
import src.indexing.vector_store as vector_store_module
from src.core.schemas import EmbeddingRecord, TextChunk
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import FaissStore


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_bm25_load_returns_false_when_chunks_change(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    chunks_path = root / "data" / "metadata" / "chunks" / "chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            TextChunk(chunk_id="paper1_deadbeef", doc_id="paper1", text="alpha baseline").model_dump(mode="json"),
            TextChunk(chunk_id="paper2_cafebabe", doc_id="paper2", text="beta result").model_dump(mode="json"),
        ],
    )

    monkeypatch.setattr(bm25_module, "find_project_root", lambda: root)
    index = BM25Index()
    index.build()

    chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + json.dumps(
        TextChunk(chunk_id="paper3_1234abcd", doc_id="paper3", text="gamma")
        .model_dump(mode="json")
    ) + "\n", encoding="utf-8")

    assert BM25Index().load() is False


def test_faiss_load_returns_false_when_embeddings_change(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    embeddings_path = root / "data" / "metadata" / "embeddings" / "embeddings.jsonl"
    _write_jsonl(
        embeddings_path,
        [
            EmbeddingRecord(
                id="paper1_deadbeef",
                text="alpha baseline",
                metadata={
                    "doc_id": "paper1",
                    "page_number": 1,
                    "headings": ["Intro"],
                    "chunk_type": "text",
                    "caption": "",
                    "asset_path": "",
                },
                embedding=[0.0, 0.1],
                content_hash="hash-1",
                embedding_model="mock-model",
            ).model_dump(mode="json"),
            EmbeddingRecord(
                id="paper2_cafebabe",
                text="beta result",
                metadata={
                    "doc_id": "paper2",
                    "page_number": 2,
                    "headings": ["Results"],
                    "chunk_type": "text",
                    "caption": "",
                    "asset_path": "",
                },
                embedding=[0.2, 0.3],
                content_hash="hash-2",
                embedding_model="mock-model",
            ).model_dump(mode="json"),
        ],
    )

    monkeypatch.setattr(vector_store_module, "find_project_root", lambda: root)
    monkeypatch.setattr(vector_store_module, "get_settings", lambda: SimpleNamespace(embedding_model="mock-model"))
    store = FaissStore(embedding_model="mock-model")
    store.build()

    embeddings_path.write_text(
        embeddings_path.read_text(encoding="utf-8")
        + json.dumps(
            EmbeddingRecord(
                id="paper3_1234abcd",
                text="gamma result",
                metadata={
                    "doc_id": "paper3",
                    "page_number": 3,
                    "headings": ["Appendix"],
                    "chunk_type": "text",
                    "caption": "",
                    "asset_path": "",
                },
                embedding=[0.4, 0.5],
                content_hash="hash-3",
                embedding_model="mock-model",
            ).model_dump(mode="json")
        )
        + "\n",
        encoding="utf-8",
    )

    reloaded = FaissStore(embedding_model="mock-model")
    assert reloaded.load() is False

