from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.ingestion.manifest import Manifest


def _write_pdf(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_should_process_new_pdf_returns_hash_and_size(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.json")
    pdf_path = _write_pdf(tmp_path / "paper1.pdf", b"alpha pdf bytes")

    decision = manifest.should_process(
        "paper1",
        pdf_path,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
    )

    assert decision.should_process is True
    assert decision.content_hash == hashlib.sha256(b"alpha pdf bytes").hexdigest()
    assert decision.file_size_bytes == len(b"alpha pdf bytes")


def test_set_complete_save_and_reload_skips_unchanged_pdf(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = Manifest(manifest_path)
    pdf_path = _write_pdf(tmp_path / "paper1.pdf", b"stable bytes")
    decision = manifest.should_process(
        "paper1",
        pdf_path,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
    )

    manifest.set_complete(
        "paper1",
        content_hash=decision.content_hash,
        file_size_bytes=decision.file_size_bytes,
        chunk_strategy="hybrid_512_50",
        num_chunks=3,
        embedding_model="model-a",
        parsed_at="2026-03-20T12:00:00Z",
        embedded_at="2026-03-20T12:05:00Z",
    )
    manifest.save()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(payload) == {"paper1"}
    assert payload["paper1"]["status"] == "complete"
    assert payload["paper1"]["embedding_model"] == "model-a"

    reloaded = Manifest(manifest_path)
    entry = reloaded.get("paper1")
    assert entry is not None
    assert entry.num_chunks == 3
    assert entry.status == "complete"

    decision_after_reload = reloaded.should_process(
        "paper1",
        pdf_path,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
    )
    assert decision_after_reload.should_process is False


def test_should_process_reingests_when_content_or_config_changes(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.json")
    pdf_path = _write_pdf(tmp_path / "paper1.pdf", b"original bytes")
    initial = manifest.should_process(
        "paper1",
        pdf_path,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
    )
    manifest.set_complete(
        "paper1",
        content_hash=initial.content_hash,
        file_size_bytes=initial.file_size_bytes,
        chunk_strategy="hybrid_512_50",
        num_chunks=2,
        embedding_model="model-a",
        parsed_at="2026-03-20T12:00:00Z",
        embedded_at="2026-03-20T12:05:00Z",
    )

    pdf_path.write_bytes(b"changed bytes")
    content_changed = manifest.should_process(
        "paper1",
        pdf_path,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
    )
    strategy_changed = manifest.should_process(
        "paper1",
        _write_pdf(tmp_path / "paper1.pdf", b"original bytes"),
        chunk_strategy="hybrid_1024_100",
        embedding_model="model-a",
    )
    model_changed = manifest.should_process(
        "paper1",
        pdf_path,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-b",
    )

    assert content_changed.should_process is True
    assert strategy_changed.should_process is True
    assert model_changed.should_process is True


def test_retry_failed_only_processes_failed_entries(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.json")
    failed_pdf = _write_pdf(tmp_path / "failed.pdf", b"failed bytes")
    complete_pdf = _write_pdf(tmp_path / "complete.pdf", b"complete bytes")

    failed_decision = manifest.should_process(
        "failed",
        failed_pdf,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
    )
    manifest.set_failed(
        "failed",
        content_hash=failed_decision.content_hash,
        file_size_bytes=failed_decision.file_size_bytes,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
        error_message="parser crashed",
    )

    complete_decision = manifest.should_process(
        "complete",
        complete_pdf,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
    )
    manifest.set_complete(
        "complete",
        content_hash=complete_decision.content_hash,
        file_size_bytes=complete_decision.file_size_bytes,
        chunk_strategy="hybrid_512_50",
        num_chunks=1,
        embedding_model="model-a",
        parsed_at="2026-03-20T12:00:00Z",
        embedded_at="2026-03-20T12:05:00Z",
    )

    retry_failed = manifest.should_process(
        "failed",
        failed_pdf,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
        retry_failed_only=True,
    )
    skip_complete = manifest.should_process(
        "complete",
        complete_pdf,
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
        retry_failed_only=True,
    )
    skip_missing = manifest.should_process(
        "missing",
        _write_pdf(tmp_path / "missing.pdf", b"missing bytes"),
        chunk_strategy="hybrid_512_50",
        embedding_model="model-a",
        retry_failed_only=True,
    )

    assert retry_failed.should_process is True
    assert skip_complete.should_process is False
    assert skip_missing.should_process is False
