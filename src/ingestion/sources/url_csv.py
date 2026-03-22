from __future__ import annotations

import csv
from pathlib import Path

from src.core.logging import get_logger
from src.ingestion.sources.base import BaseSource, DocumentMeta
from src.ingestion.sources.url_import import build_url_document, dedupe_documents, fetch_url_document

logger = get_logger(__name__)


class URLCSVSource(BaseSource):
    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(path)
        self.csv_path = self.path or self._resolve_default_input_file("*.csv", label="CSV")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"URL CSV not found: {self.csv_path}")

    def discover(self) -> list[DocumentMeta]:
        documents: list[DocumentMeta] = []
        with self.csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "url" not in fieldnames:
                raise ValueError(f"URL CSV must contain a 'url' column: {self.csv_path}")

            for index, row in enumerate(reader, start=1):
                url = str(row.get("url", "")).strip()
                if not url:
                    continue
                documents.append(
                    build_url_document(
                        url=url,
                        index=index,
                        source_type="url_csv",
                        metadata={str(key): str(value) for key, value in row.items() if key},
                    )
                )

        documents = dedupe_documents(documents)
        logger.info("url_csv_discovered", path=str(self.csv_path), count=len(documents))
        return documents

    def fetch(self, doc: DocumentMeta, dest_dir: Path) -> Path:
        return fetch_url_document(doc, dest_dir)
