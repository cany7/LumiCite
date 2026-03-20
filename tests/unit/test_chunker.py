from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from src.ingestion.chunker import extract_pdf_chunks, write_chunks_jsonl


class _FakeChunk:
    def __init__(self, text: str, heading: str | None = None, path: str = "") -> None:
        self.text = text
        self.heading = heading
        self.path = path


class _FakeConverterResult:
    def __init__(self, document: object) -> None:
        self.document = document


class _FakeDocumentConverter:
    def convert(self, pdf_path: Path) -> _FakeConverterResult:
        assert pdf_path.name == "paper1.pdf"
        return _FakeConverterResult(document=object())


class _FakeHierarchicalChunker:
    def __init__(self, min_chunk_len: int) -> None:
        self.min_chunk_len = min_chunk_len

    def chunk(self, document: object) -> list[_FakeChunk]:
        assert document is not None
        return [
            _FakeChunk("  First chunk text  ", heading="Intro", path="1/2"),
            _FakeChunk("Second chunk text", heading=None, path="3"),
            _FakeChunk("   ", heading="Ignored", path="4"),
        ]


def test_extract_pdf_chunks_builds_text_chunks(monkeypatch):
    docling_module = types.ModuleType("docling")
    document_converter_module = types.ModuleType("docling.document_converter")
    document_converter_module.DocumentConverter = _FakeDocumentConverter
    docling_core_module = types.ModuleType("docling_core")
    transforms_module = types.ModuleType("docling_core.transforms")
    chunker_module = types.ModuleType("docling_core.transforms.chunker")
    chunker_module.HierarchicalChunker = _FakeHierarchicalChunker

    monkeypatch.setitem(sys.modules, "docling", docling_module)
    monkeypatch.setitem(sys.modules, "docling.document_converter", document_converter_module)
    monkeypatch.setitem(sys.modules, "docling_core", docling_core_module)
    monkeypatch.setitem(sys.modules, "docling_core.transforms", transforms_module)
    monkeypatch.setitem(sys.modules, "docling_core.transforms.chunker", chunker_module)

    chunks = extract_pdf_chunks("paper1.pdf")

    assert len(chunks) == 2
    assert chunks[0].doc_id == "paper1"
    assert chunks[0].text == "First chunk text"
    assert chunks[0].headings == ["Intro"]
    assert chunks[0].section_path == ["1", "2"]
    assert chunks[0].source_file == "paper1.pdf"
    assert chunks[0].chunk_id.startswith("paper1_")
    assert chunks[1].section_path == ["3"]


def test_write_chunks_jsonl_writes_one_record_per_line(tmp_path: Path):
    out_path = tmp_path / "chunks.jsonl"
    chunks = extract_pdf_chunks.__globals__["TextChunk"](
        chunk_id="paper1_abc12345",
        doc_id="paper1",
        text="hello world",
        source_file="paper1.pdf",
    )

    written_path = write_chunks_jsonl([chunks], path=out_path)

    assert written_path == out_path
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["doc_id"] == "paper1"
    assert payload["text"] == "hello world"
