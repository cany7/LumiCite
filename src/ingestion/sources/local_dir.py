from __future__ import annotations

import shutil
from pathlib import Path

from src.core.logging import get_logger
from src.ingestion.sources.base import BaseSource, DocumentMeta
from src.ingestion.sources.url_import import dedupe_documents

logger = get_logger(__name__)


class LocalDirSource(BaseSource):
    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(path)
        self.directory = self.path or (self.root / "data" / "pdfs")
        if not self.directory.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.directory}")

    def discover(self) -> list[DocumentMeta]:
        documents = [
            DocumentMeta(
                doc_id=pdf_path.stem,
                source_type="local_dir",
                filename=pdf_path.name,
                local_path=pdf_path,
            )
            for pdf_path in sorted(self.directory.iterdir())
            if pdf_path.suffix.lower() == ".pdf"
        ]
        documents = dedupe_documents(documents)
        logger.info("local_dir_discovered", path=str(self.directory), count=len(documents))
        return documents

    def fetch(self, doc: DocumentMeta, dest_dir: Path) -> Path:
        if doc.local_path is None or not doc.local_path.exists():
            raise FileNotFoundError(f"Local PDF missing: {doc.filename}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / doc.filename
        if doc.local_path.resolve() != destination.resolve():
            shutil.copy2(doc.local_path, destination)
            return destination
        return doc.local_path
