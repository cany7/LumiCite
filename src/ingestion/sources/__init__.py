from __future__ import annotations

from pathlib import Path

from src.ingestion.sources.base import BaseSource, DocumentMeta
from src.ingestion.sources.local_dir import LocalDirSource
from src.ingestion.sources.url_csv import URLCSVSource
from src.ingestion.sources.url_list import URLListSource


def create_source(source_type: str, path: str | Path | None = None) -> BaseSource:
    if source_type == "url_csv":
        return URLCSVSource(path=path)

    if source_type == "local_dir":
        return LocalDirSource(path=path)
    if source_type == "url_list":
        return URLListSource(path=path)
    raise ValueError("source must be one of: local_dir, url_csv, url_list")


__all__ = [
    "BaseSource",
    "DocumentMeta",
    "LocalDirSource",
    "URLCSVSource",
    "URLListSource",
    "create_source",
]
