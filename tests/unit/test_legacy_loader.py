from __future__ import annotations

import json

from src.core.schemas import ChunkType, FigureChunk, TableChunk
from src.ingestion import legacy_loader


def test_load_grouped_dict_injects_doc_id_and_maps_fields(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            {
                "paper1": [
                    {
                        "chunkId": "paper1_tab_deadbeef",
                        "page_content": "table body",
                        "page": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    chunks = legacy_loader.load_legacy_chunks_json(path)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, TableChunk)
    assert chunk.doc_id == "paper1"
    assert chunk.chunk_id == "paper1_tab_deadbeef"
    assert chunk.text == "table body"
    assert chunk.page_number == 3


def test_load_flat_list_uses_existing_doc_id_and_derive_figure_type(tmp_path):
    path = tmp_path / "alt_text.json"
    path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "paper2_img_deadbeef",
                    "doc_id": "paper2",
                    "text": "figure alt text",
                }
            ]
        ),
        encoding="utf-8",
    )

    chunks = legacy_loader.load_legacy_chunks_json(path)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, FigureChunk)
    assert chunk.chunk_type is ChunkType.FIGURE
    assert chunk.doc_id == "paper2"


def test_invalid_chunk_is_logged_and_skipped(tmp_path, monkeypatch):
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            {
                "paper1": [
                    {"chunk_id": "paper1_deadbeef", "text": "valid text"},
                    {"chunk_id": "paper1_badc0ffe", "text": "   "},
                ]
            }
        ),
        encoding="utf-8",
    )

    warnings = []

    def fake_warning(event, **kwargs):
        warnings.append((event, kwargs))

    monkeypatch.setattr(legacy_loader.logger, "warning", fake_warning)

    chunks = legacy_loader.load_legacy_chunks_json(path)

    assert [chunk.chunk_id for chunk in chunks] == ["paper1_deadbeef"]
    assert warnings
    _event, payload = warnings[0]
    assert payload["chunk_id"] == "paper1_badc0ffe"
    assert "text must be non-empty" in payload["error"]


def test_explicit_chunk_type_takes_precedence(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "paper3_tab_deadbeef",
                    "doc_id": "paper3",
                    "text": "figure by explicit type",
                    "chunk_type": "figure",
                }
            ]
        ),
        encoding="utf-8",
    )

    chunks = legacy_loader.load_legacy_chunks_json(path)

    assert len(chunks) == 1
    assert isinstance(chunks[0], FigureChunk)


def test_unsupported_top_level_format_raises(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps("not-a-valid-top-level"), encoding="utf-8")

    try:
        legacy_loader.load_legacy_chunks_json(path)
    except ValueError as exc:
        assert "Expected dict or list" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported top-level format")
