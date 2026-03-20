from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import src.main as main_module
from src.ingestion.sources.base import DocumentMeta
from src.main import app

runner = CliRunner()


def test_ingest_dry_run_reports_discovered_documents_from_source(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    metadata_path = root / "data" / "metadata" / "metadata.csv"
    pdf_dir = root / "data" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    local_pdf = pdf_dir / "paper1.pdf"
    local_pdf.write_bytes(b"paper1")

    class FakeSource:
        def discover(self) -> list[DocumentMeta]:
            return [
                DocumentMeta(
                    doc_id="paper1",
                    source_type="metadata_csv",
                    filename="paper1.pdf",
                    local_path=local_pdf,
                ),
                DocumentMeta(
                    doc_id="paper2",
                    source_type="metadata_csv",
                    filename="paper2.pdf",
                    url="https://example.com/paper2.pdf",
                ),
            ]

    captured: dict[str, object] = {}

    def fake_create_source(source_type: str, path: Path | None) -> FakeSource:
        captured["source_type"] = source_type
        captured["path"] = path
        return FakeSource()

    monkeypatch.setattr(main_module, "find_project_root", lambda: root)
    monkeypatch.setattr(main_module, "create_source", fake_create_source)

    result = runner.invoke(
        app,
        ["ingest", "--source", "metadata_csv", "--path", str(metadata_path), "--dry-run"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert captured == {"source_type": "metadata_csv", "path": metadata_path}
    assert payload["event"] == "ingest_dry_run"
    assert payload["source"] == "metadata_csv"
    assert payload["path"] == str(metadata_path)
    assert payload["discovered"] == 2
    assert payload["files"] == [str(local_pdf), str(pdf_dir / "paper2.pdf")]
    assert payload["retry_failed"] is False
    assert payload["rebuild_index"] is False
    assert not (root / "data" / "manifest.json").exists()
