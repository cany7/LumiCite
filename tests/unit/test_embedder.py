from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from src.core.schemas import EmbeddingRecord, FigureChunk, TableChunk, TextChunk
from src.ingestion.embedder import embed_local, load_canonical_chunks_jsonl, write_embeddings_jsonl


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, cache_folder: str | None = None, **kwargs) -> None:  # noqa: ANN003
        self.model_name = model_name
        self.cache_folder = cache_folder

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):  # noqa: ANN001
        assert convert_to_numpy is True
        assert show_progress_bar is False
        return types.SimpleNamespace(tolist=lambda: [[float(index), float(index) + 0.5] for index, _ in enumerate(texts)])


def test_embed_local_uses_sentence_transformer_batches(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    vectors = embed_local(
        [
            {"id": "paper1_abc12345", "text": "alpha", "metadata": {}},
            {"id": "paper1_def67890", "text": "beta", "metadata": {}},
        ],
        model_name="mock-model",
        batch_size=1,
    )

    assert vectors == [[0.0, 0.5], [0.0, 0.5]]


def test_load_canonical_chunks_jsonl_round_trips_chunk_types(tmp_path: Path):
    path = tmp_path / "chunks.jsonl"
    records = [
        TextChunk(chunk_id="paper1_abc12345", doc_id="paper1", text="alpha"),
        TableChunk(chunk_id="paper1_tab_deadbeef", doc_id="paper1", text="table text"),
        FigureChunk(chunk_id="paper1_fig_deadbeef", doc_id="paper1", text="figure text"),
    ]
    path.write_text(
        "".join(json.dumps(record.model_dump(mode="json")) + "\n" for record in records),
        encoding="utf-8",
    )

    chunks = load_canonical_chunks_jsonl(path)

    assert isinstance(chunks[0], TextChunk)
    assert isinstance(chunks[1], TableChunk)
    assert isinstance(chunks[2], FigureChunk)


def test_write_embeddings_jsonl_writes_records(tmp_path: Path):
    path = tmp_path / "embeddings.jsonl"
    records = [
        EmbeddingRecord(
            id="paper1_abc12345",
            text="alpha",
            metadata={"doc_id": "paper1", "chunk_type": "text"},
            embedding=[0.1, 0.2],
            content_hash="hash",
            embedding_model="mock-model",
        )
    ]

    write_embeddings_jsonl(records, path)

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["id"] == "paper1_abc12345"
    assert payload["embedding_model"] == "mock-model"
