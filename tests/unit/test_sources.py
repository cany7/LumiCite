from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.ingestion.sources import create_source
from src.ingestion.sources.base import DocumentMeta
from src.ingestion.sources.local_dir import LocalDirSource
from src.ingestion.sources.metadata_csv import MetadataCSVSource
from src.ingestion.sources.url_list import URLListSource


class DummyResponse:
    def __init__(self, content: bytes = b"%PDF-1.4 mock") -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _write_metadata_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "url"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_document_meta_candidate_path_prefers_local_path(tmp_path: Path) -> None:
    local_path = tmp_path / "existing.pdf"
    local_path.write_bytes(b"pdf")
    doc = DocumentMeta(
        doc_id="paper1",
        source_type="metadata_csv",
        filename="paper1.pdf",
        local_path=local_path,
    )

    assert doc.candidate_path(tmp_path / "dest") == local_path


def test_document_meta_candidate_path_uses_destination_for_remote_docs(tmp_path: Path) -> None:
    doc = DocumentMeta(
        doc_id="paper1",
        source_type="url_list",
        filename="paper1.pdf",
        url="https://example.com/paper1.pdf",
    )

    assert doc.candidate_path(tmp_path / "dest") == tmp_path / "dest" / "paper1.pdf"


def test_create_source_dispatches_known_types(tmp_path: Path) -> None:
    metadata_csv = _write_metadata_csv(
        tmp_path / "metadata.csv",
        [{"id": "paper1", "title": "Paper 1", "url": "https://example.com/paper1.pdf"}],
    )
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper1.pdf").write_bytes(b"pdf")
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/paper2.pdf\n", encoding="utf-8")

    assert isinstance(create_source("metadata_csv", metadata_csv), MetadataCSVSource)
    assert isinstance(create_source("local_dir", pdf_dir), LocalDirSource)
    assert isinstance(create_source("url_list", url_file), URLListSource)

    with pytest.raises(ValueError, match="source must be one of"):
        create_source("unknown", tmp_path)


def test_metadata_csv_discover_skips_missing_pdfs_by_default(tmp_path: Path) -> None:
    csv_path = _write_metadata_csv(
        tmp_path / "metadata.csv",
        [
            {"id": "paper1", "title": "Paper 1", "url": "https://example.com/paper1.pdf"},
            {"id": "paper2", "title": "Paper 2", "url": "https://example.com/paper2.pdf"},
        ],
    )
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper1.pdf").write_bytes(b"pdf")

    documents = MetadataCSVSource(path=csv_path, pdf_dir=pdf_dir).discover()

    assert [doc.doc_id for doc in documents] == ["paper1"]
    assert documents[0].local_path == pdf_dir / "paper1.pdf"
    assert documents[0].title == "Paper 1"
    assert documents[0].metadata["url"] == "https://example.com/paper1.pdf"


def test_metadata_csv_discover_requires_id_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("title,url\nPaper 1,https://example.com/paper1.pdf\n", encoding="utf-8")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    source = MetadataCSVSource(path=csv_path, pdf_dir=pdf_dir)

    with pytest.raises(ValueError, match="must contain an 'id' column"):
        source.discover()


def test_metadata_csv_discover_can_include_missing_pdfs(tmp_path: Path) -> None:
    csv_path = _write_metadata_csv(
        tmp_path / "metadata.csv",
        [{"id": "paper2", "title": "Paper 2", "url": "https://example.com/paper2.pdf"}],
    )
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    documents = MetadataCSVSource(path=csv_path, pdf_dir=pdf_dir, include_missing=True).discover()

    assert len(documents) == 1
    assert documents[0].doc_id == "paper2"
    assert documents[0].local_path is None
    assert documents[0].url == "https://example.com/paper2.pdf"


def test_metadata_csv_fetch_returns_existing_local_pdf_without_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = _write_metadata_csv(
        tmp_path / "metadata.csv",
        [{"id": "paper1", "title": "Paper 1", "url": "https://example.com/paper1.pdf"}],
    )
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    local_pdf = pdf_dir / "paper1.pdf"
    local_pdf.write_bytes(b"existing pdf")

    source = MetadataCSVSource(path=csv_path, pdf_dir=pdf_dir)
    document = source.discover()[0]

    monkeypatch.setattr(
        "src.ingestion.sources.metadata_csv.requests.get",
        lambda *args, **kwargs: pytest.fail("requests.get should not be called for local PDFs"),
    )

    assert source.fetch(document, tmp_path / "downloads") == local_pdf


def test_metadata_csv_fetch_downloads_missing_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = _write_metadata_csv(
        tmp_path / "metadata.csv",
        [{"id": "paper2", "title": "Paper 2", "url": "https://example.com/paper2.pdf"}],
    )
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    source = MetadataCSVSource(path=csv_path, pdf_dir=pdf_dir, include_missing=True)
    document = source.discover()[0]

    monkeypatch.setattr(
        "src.ingestion.sources.metadata_csv.requests.get",
        lambda url, timeout: DummyResponse(b"downloaded metadata pdf"),
    )

    fetched_path = source.fetch(document, tmp_path / "downloads")

    assert fetched_path == tmp_path / "downloads" / "paper2.pdf"
    assert fetched_path.read_bytes() == b"downloaded metadata pdf"


def test_local_dir_source_discovers_sorted_pdfs_only(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "b-paper.pdf").write_bytes(b"pdf-b")
    (pdf_dir / "a-paper.pdf").write_bytes(b"pdf-a")
    (pdf_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    documents = LocalDirSource(path=pdf_dir).discover()

    assert [doc.doc_id for doc in documents] == ["a-paper", "b-paper"]
    assert all(doc.local_path is not None for doc in documents)


def test_url_list_source_discovers_filenames_and_fetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "https://example.com/papers/paper1.pdf\nhttps://example.com/download/paper2\n\n",
        encoding="utf-8",
    )
    source = URLListSource(path=url_file)

    documents = source.discover()
    assert [doc.filename for doc in documents] == ["paper1.pdf", "paper2.pdf"]
    assert [doc.doc_id for doc in documents] == ["paper1", "paper2"]

    monkeypatch.setattr(
        "src.ingestion.sources.url_list.requests.get",
        lambda url, timeout: DummyResponse(b"url list pdf"),
    )

    fetched_path = source.fetch(documents[1], tmp_path / "downloads")

    assert fetched_path == tmp_path / "downloads" / "paper2.pdf"
    assert fetched_path.read_bytes() == b"url list pdf"
