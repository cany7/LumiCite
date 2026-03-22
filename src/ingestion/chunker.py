from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import get_settings
from src.core.logging import get_logger
from src.core.paths import chunks_jsonl_path, find_project_root
from src.core.schemas import ChunkModel, ChunkType, TextChunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class TextBlock:
    text: str
    page_number: int | None
    headings: list[str]


def build_chunk_id(doc_id: str, chunk_type: ChunkType, seed: str) -> str:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
    if chunk_type == ChunkType.FIGURE:
        return f"{doc_id}_fig_{digest}"
    if chunk_type == ChunkType.TABLE:
        return f"{doc_id}_tab_{digest}"
    return f"{doc_id}_{digest}"


def _split_long_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    pieces: list[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        end = min(text_len, start + chunk_size)
        if end < text_len:
            split = normalized.rfind(" ", start + max(1, chunk_size // 2), end)
            if split <= start:
                split = end
        else:
            split = end

        piece = normalized[start:split].strip()
        if piece:
            pieces.append(piece)
        if split >= text_len:
            break
        start = max(split - chunk_overlap, start + 1)

    return pieces


def split_text_blocks(
    doc_id: str,
    blocks: list[TextBlock],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    settings = get_settings()
    max_chars = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    grouped: list[tuple[str, int | None, list[str]]] = []
    buffer_parts: list[str] = []
    buffer_start_page: int | None = None
    buffer_headings: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer_parts, buffer_start_page, buffer_headings
        if not buffer_parts:
            return
        text = " ".join(part.strip() for part in buffer_parts if part.strip()).strip()
        if text:
            grouped.append((text, buffer_start_page, list(buffer_headings)))
        buffer_parts = []
        buffer_start_page = None
        buffer_headings = []

    for block in blocks:
        text = " ".join(block.text.split()).strip()
        if not text:
            continue

        if not buffer_parts:
            buffer_start_page = block.page_number
            buffer_headings = list(block.headings)
        elif block.headings != buffer_headings:
            flush_buffer()
            buffer_start_page = block.page_number
            buffer_headings = list(block.headings)
        elif buffer_start_page is None and block.page_number is not None:
            buffer_start_page = block.page_number

        buffer_parts.append(text)
        current_text = " ".join(buffer_parts)
        if len(current_text) > max_chars:
            flush_buffer()

    flush_buffer()

    chunks: list[TextChunk] = []
    for text, page_number, headings in grouped:
        for piece in _split_long_text(text, chunk_size=max_chars, chunk_overlap=overlap):
            chunk_id = build_chunk_id(
                doc_id,
                ChunkType.TEXT,
                f"{page_number}|{'/'.join(headings)}|{piece}",
            )
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=piece,
                    page_number=page_number,
                    headings=list(headings),
                )
            )

    logger.info("text_chunks_built", doc_id=doc_id, num_chunks=len(chunks))
    return chunks


def write_chunks_jsonl(chunks: list[ChunkModel], path: str | Path | None = None) -> Path:
    root = find_project_root()
    out_path = Path(path) if path else chunks_jsonl_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        chunks,
        key=lambda chunk: (
            chunk.doc_id,
            chunk.page_number or 0,
            chunk.chunk_type.value,
            chunk.chunk_id,
        ),
    )
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in ordered:
            handle.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")

    logger.info("chunks_jsonl_written", path=str(out_path), num_chunks=len(ordered))
    return out_path
