from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.ingestion.sources import create_source
from src.ingestion.sources.local_dir import LocalDirSource
from src.ingestion.sources.url_csv import URLCSVSource
from src.ingestion.sources.url_list import URLListSource


class DummyResponse:
    def __init__(self, content: bytes = b"%PDF-1.4 mock") -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _write_url_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "url"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_create_source_dispatches_known_types(tmp_path: Path) -> None:
    csv_path = _write_url_csv(
        tmp_path / "papers.csv",
        [{"id": "ignored", "title": "Paper 1", "url": "https://example.com/paper1.pdf"}],
    )
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper1.pdf").write_bytes(b"pdf")
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/paper2.pdf\n", encoding="utf-8")

    assert isinstance(create_source("url_csv", csv_path), URLCSVSource)
    assert isinstance(create_source("local_dir", pdf_dir), LocalDirSource)
    assert isinstance(create_source("url_list", url_file), URLListSource)

    with pytest.raises(ValueError, match="source must be one of"):
        create_source("unknown", tmp_path)


def test_url_csv_discovers_from_url_column_and_ignores_id_identity(tmp_path: Path) -> None:
    csv_path = _write_url_csv(
        tmp_path / "papers.csv",
        [{"id": "custom-id", "title": "Paper 1", "url": "https://example.com/files/paper1.pdf"}],
    )

    documents = URLCSVSource(path=csv_path).discover()

    assert len(documents) == 1
    assert documents[0].doc_id == "paper1"
    assert documents[0].source_type == "url_csv"
    assert documents[0].metadata["id"] == "custom-id"


def test_url_csv_dedupes_duplicate_doc_ids(tmp_path: Path) -> None:
    csv_path = _write_url_csv(
        tmp_path / "papers.csv",
        [
            {"id": "one", "title": "Paper 1", "url": "https://example.com/a/paper1.pdf"},
            {"id": "two", "title": "Paper 1 copy", "url": "https://example.com/b/paper1.pdf"},
        ],
    )

    documents = URLCSVSource(path=csv_path).discover()

    assert [document.doc_id for document in documents] == ["paper1"]


def test_url_csv_requires_url_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "papers.csv"
    csv_path.write_text("title\nPaper 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a 'url' column"):
        URLCSVSource(path=csv_path).discover()


def test_url_csv_auto_discovers_single_csv_under_data_pdfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    csv_path = _write_url_csv(
        root / "data" / "pdfs" / "sample_urls.csv",
        [{"id": "one", "title": "Paper 1", "url": "https://example.com/files/paper1.pdf"}],
    )
    monkeypatch.setattr("src.ingestion.sources.base.find_project_root", lambda: root)

    source = URLCSVSource()

    assert source.csv_path == csv_path


def test_url_list_auto_discovery_requires_path_when_multiple_txt_files_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    pdf_dir = root / "data" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / "a.txt").write_text("https://example.com/a.pdf\n", encoding="utf-8")
    (pdf_dir / "b.txt").write_text("https://example.com/b.pdf\n", encoding="utf-8")
    monkeypatch.setattr("src.ingestion.sources.base.find_project_root", lambda: root)

    with pytest.raises(FileNotFoundError, match="Multiple TXT files found"):
        URLListSource()


def test_local_dir_fetch_copies_into_canonical_pdf_dir(tmp_path: Path) -> None:
    source_dir = tmp_path / "source-pdfs"
    source_dir.mkdir()
    local_pdf = source_dir / "paper1.pdf"
    local_pdf.write_bytes(b"original")
    source = LocalDirSource(path=source_dir)
    document = source.discover()[0]

    fetched_path = source.fetch(document, tmp_path / "data" / "pdfs")

    assert fetched_path == tmp_path / "data" / "pdfs" / "paper1.pdf"
    assert fetched_path.read_bytes() == b"original"


def test_url_list_discovers_filenames_and_fetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url_file = tmp_path / "papers.txt"
    url_file.write_text(
        "https://example.com/papers/paper1.pdf\nhttps://example.com/download/paper2\n",
        encoding="utf-8",
    )
    source = URLListSource(path=url_file)
    documents = source.discover()

    monkeypatch.setattr(
        "src.ingestion.sources.url_import.requests.get",
        lambda url, timeout: DummyResponse(b"url list pdf"),
    )

    fetched_path = source.fetch(documents[1], tmp_path / "downloads")

    assert [doc.filename for doc in documents] == ["paper1.pdf", "paper2.pdf"]
    assert [doc.doc_id for doc in documents] == ["paper1", "paper2"]
    assert fetched_path == tmp_path / "downloads" / "paper2.pdf"
    assert fetched_path.read_bytes() == b"url list pdf"


def test_fetch_url_document_skips_download_when_pdf_already_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url_file = tmp_path / "papers.txt"
    url_file.write_text("https://example.com/papers/paper1.pdf\n", encoding="utf-8")
    source = URLListSource(path=url_file)
    document = source.discover()[0]
    destination_dir = tmp_path / "data" / "pdfs"
    destination_dir.mkdir(parents=True, exist_ok=True)
    existing_pdf = destination_dir / "paper1.pdf"
    existing_pdf.write_bytes(b"existing pdf")

    def fail_get(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("requests.get should not be called when the target PDF already exists")

    monkeypatch.setattr("src.ingestion.sources.url_import.requests.get", fail_get)

    fetched_path = source.fetch(document, destination_dir)

    assert fetched_path == existing_pdf
    assert fetched_path.read_bytes() == b"existing pdf"
