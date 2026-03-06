from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.core.logging import get_logger
from src.core.schemas import ChunkType, FigureChunk, TableChunk, TextChunk

logger = get_logger(__name__)

ChunkModel = TextChunk | TableChunk | FigureChunk


def _normalize_legacy_fields(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    if "chunkId" in normalized and "chunk_id" not in normalized:
        normalized["chunk_id"] = normalized.pop("chunkId")
    if "page_content" in normalized and "text" not in normalized:
        normalized["text"] = normalized.pop("page_content")
    if "page" in normalized and "page_number" not in normalized:
        normalized["page_number"] = normalized.pop("page")
    return normalized


def _derive_chunk_type(chunk: dict[str, Any]) -> ChunkType:
    explicit = chunk.get("chunk_type")
    if explicit:
        return ChunkType(str(explicit))

    chunk_id = str(chunk.get("chunk_id", "")).lower()
    if "_img_" in chunk_id or "_fig_" in chunk_id:
        return ChunkType.FIGURE
    if "_tab_" in chunk_id:
        return ChunkType.TABLE
    return ChunkType.TEXT


def _build_model(chunk: dict[str, Any]) -> ChunkModel:
    chunk_type = _derive_chunk_type(chunk)
    chunk["chunk_type"] = chunk_type
    if chunk_type == ChunkType.TABLE:
        return TableChunk(**chunk)
    if chunk_type == ChunkType.FIGURE:
        return FigureChunk(**chunk)
    return TextChunk(**chunk)


def _iter_raw_chunks(data: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if isinstance(data, dict):
        for doc_id, chunks in data.items():
            if not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                entry = _normalize_legacy_fields(chunk)
                entry["doc_id"] = str(entry.get("doc_id") or doc_id)
                records.append(entry)
        return records

    if isinstance(data, list):
        for chunk in data:
            if not isinstance(chunk, dict):
                continue
            entry = _normalize_legacy_fields(chunk)
            entry["doc_id"] = str(entry.get("doc_id") or "")
            records.append(entry)
        return records

    raise ValueError("Unsupported legacy chunk format. Expected dict or list at top-level.")


def load_legacy_chunks_json(path: str | Path) -> list[ChunkModel]:
    legacy_path = Path(path)
    raw_data = json.loads(legacy_path.read_text(encoding="utf-8"))

    valid_chunks: list[ChunkModel] = []
    for record in _iter_raw_chunks(raw_data):
        chunk_id = str(record.get("chunk_id") or record.get("id") or "unknown")
        try:
            valid_chunks.append(_build_model(record))
        except (ValidationError, ValueError) as exc:
            logger.warning("legacy_chunk_skipped", chunk_id=chunk_id, error=str(exc))

    return valid_chunks
