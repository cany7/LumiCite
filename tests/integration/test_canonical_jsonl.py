from __future__ import annotations

import json
from pathlib import Path

from src.core.schemas import EmbeddingRecord, FigureChunk, TextChunk
from src.ingestion import json_builder
from src.ingestion.chunker import write_chunks_jsonl
from src.ingestion.embedder import load_ingestion_chunks, write_embeddings_jsonl


def test_write_chunks_jsonl_emits_one_json_record_per_line(tmp_path: Path):
    out_path = tmp_path / "chunks.jsonl"
    chunks = [
        TextChunk(chunk_id="paper1_aaaabbbb", doc_id="paper1", text="alpha"),
        FigureChunk(chunk_id="paper1_fig_deadbeef", doc_id="paper1", text="beta"),
    ]

    written_path = write_chunks_jsonl(chunks, path=out_path)

    lines = written_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["chunk_id"] == "paper1_aaaabbbb"
    assert json.loads(lines[1])["chunk_type"] == "figure"


def test_write_embeddings_jsonl_emits_one_json_record_per_line(tmp_path: Path):
    out_path = tmp_path / "embeddings.jsonl"
    records = [
        EmbeddingRecord(
            id="paper1_aaaabbbb",
            text="alpha",
            metadata={"doc_id": "paper1", "chunk_type": "text"},
            embedding=[0.1, 0.2],
            content_hash="2c1743a391305fbf367df8e4f069f9f9",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
    ]

    write_embeddings_jsonl(records, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == "paper1_aaaabbbb"
    assert payload["metadata"]["doc_id"] == "paper1"


def test_load_ingestion_chunks_prefers_canonical_jsonl_over_legacy(json_dir: Path):
    canonical_path = json_dir / "chunks.jsonl"
    canonical_path.write_text(
        json.dumps(
            {
                "chunk_id": "paper1_deadbeef",
                "doc_id": "paper1",
                "text": "canonical text",
                "chunk_type": "text",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (json_dir / "chunks.json").write_text(
        json.dumps({"paper1": [{"chunk_id": "paper1_legacycafe", "text": "legacy text"}]}),
        encoding="utf-8",
    )

    chunks = load_ingestion_chunks(json_dir)

    assert [chunk.chunk_id for chunk in chunks] == ["paper1_deadbeef"]


def test_load_ingestion_chunks_falls_back_to_legacy_when_jsonl_missing(json_dir: Path):
    (json_dir / "alt_text.json").write_text(
        json.dumps(
            [
                {
                    "chunk_id": "paper2_fig_deadbeef",
                    "doc_id": "paper2",
                    "text": "legacy figure text",
                }
            ]
        ),
        encoding="utf-8",
    )

    chunks = load_ingestion_chunks(json_dir)

    assert len(chunks) == 1
    assert chunks[0].doc_id == "paper2"
    assert chunks[0].chunk_type.value == "figure"


def test_json_builder_writes_canonical_chunks_jsonl_not_legacy_chunks_json(tmp_path, monkeypatch):
    monkeypatch.setattr(json_builder, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(json_builder, "load_metadata_df", lambda: {"id": ["paper1"]})
    monkeypatch.setattr(json_builder, "get_PDF_paths", lambda: {"paper1": "/tmp/paper1.pdf"})
    monkeypatch.setattr(
        json_builder,
        "extract_pdf_chunks",
        lambda _path: [TextChunk(chunk_id="paper1_deadbeef", doc_id="paper1", text="alpha")],
    )

    out_path = json_builder.build_jsonl()

    assert out_path == tmp_path / "data" / "JSON" / "chunks.jsonl"
    assert out_path.exists()
    assert not (tmp_path / "data" / "JSON" / "chunks.json").exists()
