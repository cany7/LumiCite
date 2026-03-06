from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.config.settings import get_settings
from src.core.logging import get_logger
from src.core.paths import find_project_root
from src.core.schemas import TextChunk

logger = get_logger(__name__)


def extract_pdf_chunks(pdf_path: str | Path) -> list[TextChunk]:
    pdf_path = Path(pdf_path)
    settings = get_settings()
    from docling_core.transforms.chunker import HierarchicalChunker
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    document = result.document

    chunker = HierarchicalChunker(min_chunk_len=max(1, settings.chunk_overlap))

    chunks: list[TextChunk] = []
    doc_id = pdf_path.stem
    for chunk in chunker.chunk(document):
        text = str(getattr(chunk, "text", "")).strip()
        if not text:
            continue

        heading = getattr(chunk, "heading", None)
        headings = [str(heading)] if heading else []
        section_path = [seg for seg in str(getattr(chunk, "path", "")).split("/") if seg]

        chunks.append(
            TextChunk(
                chunk_id=f"{doc_id}_{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                text=text,
                page_number=None,
                section_path=section_path,
                headings=[str(h) for h in headings if h],
                source_file=pdf_path.name,
            )
        )

    logger.info("chunks_extracted", source_file=pdf_path.name, num_chunks=len(chunks))
    return chunks


def write_chunks_jsonl(chunks: list[TextChunk], path: str | Path | None = None) -> Path:
    root = find_project_root()
    out_path = Path(path) if path else root / "data" / "JSON" / "chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

    logger.info("chunks_jsonl_written", path=str(out_path), num_chunks=len(chunks))
    return out_path
