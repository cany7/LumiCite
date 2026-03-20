from __future__ import annotations

import csv
from pathlib import Path

import requests

from src.core.logging import get_logger
from src.ingestion.sources.base import BaseSource, DocumentMeta

logger = get_logger(__name__)


class MetadataCSVSource(BaseSource):
    """Discover PDFs from metadata rows keyed by the canonical `id` column."""

    def __init__(
        self,
        path: str | Path | None = None,
        pdf_dir: str | Path | None = None,
        include_missing: bool = False,
    ) -> None:
        super().__init__(path)
        self.csv_path = self.path or (self.root / "data" / "metadata" / "metadata.csv")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.csv_path}")
        self.pdf_dir = self._resolve_path(pdf_dir) if pdf_dir is not None else (self.root / "data" / "pdfs")
        self.include_missing = include_missing

    def discover(self) -> list[DocumentMeta]:
        documents: list[DocumentMeta] = []
        with self.csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "id" not in (reader.fieldnames or []):
                raise ValueError(f"Metadata CSV must contain an 'id' column: {self.csv_path}")
            for row in reader:
                doc_id = str(row.get("id", "")).strip()
                if not doc_id:
                    continue

                local_path = self.pdf_dir / f"{doc_id}.pdf"
                if not local_path.exists() and not self.include_missing:
                    continue

                documents.append(
                    DocumentMeta(
                        doc_id=doc_id,
                        source_type="metadata_csv",
                        filename=f"{doc_id}.pdf",
                        local_path=local_path if local_path.exists() else None,
                        url=str(row.get("url", "")).strip(),
                        title=str(row.get("title", "")).strip(),
                        metadata=dict(row),
                    )
                )

        logger.info("metadata_csv_discovered", path=str(self.csv_path), count=len(documents))
        return documents

    def fetch(self, doc: DocumentMeta, dest_dir: Path) -> Path:
        if doc.local_path is not None and doc.local_path.exists():
            return doc.local_path

        if not doc.url:
            raise FileNotFoundError(f"No local PDF or URL available for document: {doc.doc_id}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        target_path = dest_dir / doc.filename
        response = requests.get(doc.url, timeout=60)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        logger.info("metadata_csv_downloaded", doc_id=doc.doc_id, path=str(target_path))
        return target_path
