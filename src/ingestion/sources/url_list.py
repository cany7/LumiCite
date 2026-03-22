from __future__ import annotations

from pathlib import Path

from src.core.logging import get_logger
from src.ingestion.sources.base import BaseSource, DocumentMeta
from src.ingestion.sources.url_import import build_url_document, dedupe_documents, fetch_url_document

logger = get_logger(__name__)


class URLListSource(BaseSource):
    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(path)
        self.url_file = self.path or self._resolve_default_input_file("*.txt", label="TXT")
        if self.url_file is None or not self.url_file.exists():
            raise FileNotFoundError("URL list file is required for --source url_list")

    def discover(self) -> list[DocumentMeta]:
        assert self.url_file is not None
        documents: list[DocumentMeta] = []
        for index, line in enumerate(self.url_file.read_text(encoding="utf-8").splitlines(), start=1):
            url = line.strip()
            if not url:
                continue
            documents.append(build_url_document(url=url, index=index, source_type="url_list"))

        documents = dedupe_documents(documents)
        logger.info("url_list_discovered", path=str(self.url_file), count=len(documents))
        return documents

    def fetch(self, doc: DocumentMeta, dest_dir: Path) -> Path:
        return fetch_url_document(doc, dest_dir)
