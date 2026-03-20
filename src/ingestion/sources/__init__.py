from __future__ import annotations

from pathlib import Path

from src.ingestion.sources.base import BaseSource, DocumentMeta
from src.ingestion.sources.local_dir import LocalDirSource
from src.ingestion.sources.metadata_csv import MetadataCSVSource
from src.ingestion.sources.url_list import URLListSource


def create_source(source_type: str, path: str | Path | None = None) -> BaseSource:
    if source_type == "metadata_csv":
        return MetadataCSVSource(path=path)

    if source_type == "local_dir":
        return LocalDirSource(path=path)
    if source_type == "url_list":
        return URLListSource(path=path)
    raise ValueError("source must be one of: metadata_csv, local_dir, url_list")


__all__ = [
    "BaseSource",
    "DocumentMeta",
    "LocalDirSource",
    "MetadataCSVSource",
    "URLListSource",
    "create_source",
]
