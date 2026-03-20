from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.core.schemas import ManifestEntry


def _now_utc() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parser_version() -> str:
    try:
        return f"docling-{version('docling')}"
    except PackageNotFoundError:
        return "docling-unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestDecision:
    should_process: bool
    action: str
    content_hash: str
    file_size_bytes: int


class Manifest:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, ManifestEntry] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = {}
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest.json must be a JSON object keyed by doc_id")

        self.entries = {
            str(doc_id): ManifestEntry(**entry)
            for doc_id, entry in payload.items()
            if isinstance(entry, dict)
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload = {
            doc_id: entry.model_dump(mode="json")
            for doc_id, entry in sorted(self.entries.items())
        }
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, self.path)

    def get(self, doc_id: str) -> ManifestEntry | None:
        return self.entries.get(doc_id)

    def should_process(
        self,
        doc_id: str,
        pdf_path: Path,
        *,
        chunk_strategy: str,
        embedding_model: str,
        retry_failed_only: bool = False,
    ) -> ManifestDecision:
        content_hash = _file_sha256(pdf_path)
        file_size_bytes = pdf_path.stat().st_size
        entry = self.entries.get(doc_id)

        if retry_failed_only:
            if entry is None or entry.status != "failed":
                return ManifestDecision(False, "skip_non_failed", content_hash, file_size_bytes)
            return ManifestDecision(True, "retry_failed", content_hash, file_size_bytes)

        if entry is None:
            return ManifestDecision(True, "new", content_hash, file_size_bytes)
        if entry.status == "failed":
            return ManifestDecision(True, "previous_failed", content_hash, file_size_bytes)
        if entry.content_hash != content_hash:
            return ManifestDecision(True, "content_changed", content_hash, file_size_bytes)
        if entry.chunk_strategy != chunk_strategy:
            return ManifestDecision(True, "chunk_strategy_changed", content_hash, file_size_bytes)
        if entry.embedding_model != embedding_model:
            return ManifestDecision(True, "embedding_model_changed", content_hash, file_size_bytes)
        return ManifestDecision(False, "skipped", content_hash, file_size_bytes)

    def set_complete(
        self,
        doc_id: str,
        *,
        content_hash: str,
        file_size_bytes: int,
        chunk_strategy: str,
        num_chunks: int,
        embedding_model: str,
        parsed_at: str | None = None,
        embedded_at: str | None = None,
    ) -> None:
        parsed_at = parsed_at or _now_utc()
        embedded_at = embedded_at or parsed_at
        self.entries[doc_id] = ManifestEntry(
            content_hash=content_hash,
            file_size_bytes=file_size_bytes,
            parsed_at=parsed_at,
            parser_version=_parser_version(),
            chunk_strategy=chunk_strategy,
            num_chunks=num_chunks,
            embedding_model=embedding_model,
            embedded_at=embedded_at,
            status="complete",
            error_message="",
        )

    def set_failed(
        self,
        doc_id: str,
        *,
        content_hash: str,
        file_size_bytes: int,
        chunk_strategy: str,
        embedding_model: str,
        error_message: str,
    ) -> None:
        timestamp = _now_utc()
        self.entries[doc_id] = ManifestEntry(
            content_hash=content_hash,
            file_size_bytes=file_size_bytes,
            parsed_at=timestamp,
            parser_version=_parser_version(),
            chunk_strategy=chunk_strategy,
            num_chunks=0,
            embedding_model=embedding_model,
            embedded_at=timestamp,
            status="failed",
            error_message=error_message,
        )
