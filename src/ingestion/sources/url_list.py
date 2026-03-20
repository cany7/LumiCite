from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests

from src.core.logging import get_logger
from src.ingestion.sources.base import BaseSource, DocumentMeta

logger = get_logger(__name__)


class URLListSource(BaseSource):
    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(path)
        self.url_file = self.path
        if self.url_file is None or not self.url_file.exists():
            raise FileNotFoundError("URL list file is required for --source url_list")

    def discover(self) -> list[DocumentMeta]:
        assert self.url_file is not None
        documents: list[DocumentMeta] = []
        for index, line in enumerate(self.url_file.read_text(encoding="utf-8").splitlines(), start=1):
            url = line.strip()
            if not url:
                continue
            parsed = urlparse(url)
            filename = Path(parsed.path).name or f"url_{index}.pdf"
            if not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"

            documents.append(
                DocumentMeta(
                    doc_id=Path(filename).stem,
                    source_type="url_list",
                    filename=filename,
                    url=url,
                )
            )

        logger.info("url_list_discovered", path=str(self.url_file), count=len(documents))
        return documents

    def fetch(self, doc: DocumentMeta, dest_dir: Path) -> Path:
        if not doc.url:
            raise FileNotFoundError(f"No URL available for document: {doc.doc_id}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        target_path = dest_dir / doc.filename
        response = requests.get(doc.url, timeout=60)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        logger.info("url_list_downloaded", doc_id=doc.doc_id, path=str(target_path))
        return target_path
