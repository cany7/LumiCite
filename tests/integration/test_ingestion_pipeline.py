from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.ingestion.pipeline as pipeline_module
from src.core.schemas import EmbeddingRecord, TextChunk
from src.ingestion.manifest import Manifest
from src.ingestion.pipeline import ProcessedDocument, run_ingest
from src.ingestion.sources.base import DocumentMeta


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_chunks(path: Path, chunks: list[TextChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )


def test_run_ingest_dry_run_reports_new_and_prune_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    existing_chunk = TextChunk(chunk_id="stale_deadbeef", doc_id="stale", text="old evidence")
    _write_chunks(root / "data" / "metadata" / "chunks" / "chunks.jsonl", [existing_chunk])

    manifest = Manifest(root / "data" / "manifest.json")
    manifest.set_complete(
        "stale",
        content_hash="old-hash",
        file_size_bytes=10,
        num_chunks=1,
        embedding_model="mock-model",
        parsed_at="2026-03-21T10:00:00Z",
        embedded_at="2026-03-21T10:01:00Z",
    )
    manifest.save()

    source_pdf = root / "fixtures" / "new-paper.pdf"
    source_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf.write_bytes(b"%PDF-1.4 new paper")
    documents = [
        DocumentMeta(
            doc_id="new-paper",
            source_type="local_dir",
            filename="new-paper.pdf",
            local_path=source_pdf,
        )
    ]

    class FakeSource:
        def discover(self) -> list[DocumentMeta]:
            return documents

        def fetch(self, doc: DocumentMeta, dest_dir: Path) -> Path:
            dest_dir.mkdir(parents=True, exist_ok=True)
            destination = dest_dir / doc.filename
            destination.write_bytes(source_pdf.read_bytes())
            return destination

    monkeypatch.setattr(pipeline_module, "find_project_root", lambda: root)
    monkeypatch.setattr(
        pipeline_module,
        "get_settings",
        lambda: SimpleNamespace(embedding_model="mock-model", embedding_batch_size=8),
    )
    monkeypatch.setattr(pipeline_module, "create_source", lambda source, path: FakeSource())

    summary = run_ingest(source="local_dir", dry_run=True)

    assert summary["event"] == "parse_dry_run"
    assert {"doc_id": "stale", "action": "prune"} in summary["planned_actions"]
    assert any(action["doc_id"] == "new-paper" and action["action"] == "new" for action in summary["planned_actions"])
    assert "stale_deadbeef" in (root / "data" / "metadata" / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")


def test_run_ingest_prunes_stale_docs_and_drops_failed_reingest_from_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    chunks_path = root / "data" / "metadata" / "chunks" / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            TextChunk(chunk_id="stale_deadbeef", doc_id="stale", text="stale evidence"),
            TextChunk(chunk_id="badpaper_deadbeef", doc_id="badpaper", text="old bad evidence"),
        ],
    )

    manifest = Manifest(root / "data" / "manifest.json")
    manifest.set_complete(
        "stale",
        content_hash="stale-hash",
        file_size_bytes=10,
        num_chunks=1,
        embedding_model="mock-model",
        parsed_at="2026-03-21T10:00:00Z",
        embedded_at="2026-03-21T10:01:00Z",
    )
    manifest.set_complete(
        "badpaper",
        content_hash="old-hash",
        file_size_bytes=10,
        num_chunks=1,
        embedding_model="mock-model",
        parsed_at="2026-03-21T10:00:00Z",
        embedded_at="2026-03-21T10:01:00Z",
    )
    manifest.save()

    fixtures = root / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    keep_pdf = fixtures / "keep.pdf"
    bad_pdf = fixtures / "badpaper.pdf"
    keep_pdf.write_bytes(b"%PDF-1.4 keep")
    bad_pdf.write_bytes(b"%PDF-1.4 changed badpaper")
    docs = [
        DocumentMeta(doc_id="keep", source_type="local_dir", filename="keep.pdf", local_path=keep_pdf),
        DocumentMeta(doc_id="badpaper", source_type="local_dir", filename="badpaper.pdf", local_path=bad_pdf),
    ]

    class FakeSource:
        def discover(self) -> list[DocumentMeta]:
            return docs

        def fetch(self, doc: DocumentMeta, dest_dir: Path) -> Path:
            dest_dir.mkdir(parents=True, exist_ok=True)
            destination = dest_dir / doc.filename
            destination.write_bytes(doc.local_path.read_bytes())  # type: ignore[arg-type]
            return destination

    def fake_run_jobs(root_arg: Path, jobs, device: str, llm_backend: str | None):  # noqa: ANN001
        decisions = {job.doc_id: job.decision for job in jobs}
        assert llm_backend == "api"
        return [
            ProcessedDocument(
                doc_id="keep",
                content_hash=decisions["keep"].content_hash,
                file_size_bytes=decisions["keep"].file_size_bytes,
                chunks=[TextChunk(chunk_id="keep_deadbeef", doc_id="keep", text="fresh keep evidence")],
            ),
            ProcessedDocument(
                doc_id="badpaper",
                content_hash=decisions["badpaper"].content_hash,
                file_size_bytes=decisions["badpaper"].file_size_bytes,
                chunks=[],
                error="mineru crashed",
                error_type="ingest_failed",
            ),
        ]

    def fake_build_embedding_records(chunks, *, embedding_model: str, batch_size: int):  # noqa: ANN001
        return [
            EmbeddingRecord(
                id=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "doc_id": chunk.doc_id,
                    "page_number": chunk.page_number,
                    "headings": list(chunk.headings),
                    "chunk_type": chunk.chunk_type.value,
                    "caption": getattr(chunk, "caption", ""),
                    "asset_path": getattr(chunk, "asset_path", ""),
                },
                embedding=[0.1, 0.2],
                content_hash="embed-hash",
                embedding_model=embedding_model,
                created_at="2026-03-21T10:02:00Z",
            )
            for chunk in chunks
        ]

    monkeypatch.setattr(pipeline_module, "find_project_root", lambda: root)
    monkeypatch.setattr(
        pipeline_module,
        "get_settings",
        lambda: SimpleNamespace(embedding_model="mock-model", embedding_batch_size=8),
    )
    monkeypatch.setattr(pipeline_module, "create_source", lambda source, path: FakeSource())
    monkeypatch.setattr(pipeline_module, "_run_jobs", fake_run_jobs)
    monkeypatch.setattr(pipeline_module, "build_embedding_records", fake_build_embedding_records)
    monkeypatch.setattr(
        pipeline_module,
        "_build_search_indices",
        lambda **kwargs: {
            "faiss_index": str(root / "data" / "metadata" / "faiss" / "my_faiss.index"),
            "faiss_metadata": str(root / "data" / "metadata" / "faiss" / "my_faiss.meta.json"),
            "text_data": str(root / "data" / "metadata" / "faiss" / "text_data.pkl"),
            "vectors": 1,
            "bm25_index": str(root / "data" / "metadata" / "bm25" / "bm25_index.pkl"),
            "bm25_metadata": str(root / "data" / "metadata" / "bm25" / "bm25_index.meta.json"),
            "bm25_rows": 1,
        },
    )

    summary = run_ingest(source="local_dir")

    written_chunks = (root / "data" / "metadata" / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    saved_manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))

    assert summary["status"] == "partial"
    assert summary["processed_pdfs"] == 1
    assert summary["failed_pdfs"] == 1
    assert summary["pruned_pdfs"] == 1
    assert summary["chunks_written"] == 1
    assert "keep_deadbeef" in written_chunks
    assert "stale_deadbeef" not in written_chunks
    assert "badpaper_deadbeef" not in written_chunks
    assert "stale" not in saved_manifest
    assert saved_manifest["keep"]["status"] == "complete"
    assert saved_manifest["badpaper"]["status"] == "failed"
    assert saved_manifest["badpaper"]["content_hash"] == _sha256(b"%PDF-1.4 changed badpaper")
