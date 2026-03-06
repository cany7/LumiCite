from __future__ import annotations

import json
from pathlib import Path

from src.core.logging import get_logger
from src.core.paths import find_project_root
from src.ingestion.chunker import extract_pdf_chunks
from src.ingestion.load_files import get_PDF_paths, load_metadata_df

logger = get_logger(__name__)


def canonical_chunks_path() -> Path:
    root = find_project_root()
    json_dir = root / "data" / "JSON"
    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir / "chunks.jsonl"


def write_chunks_jsonl(chunks: list[dict], path: Path) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def build_jsonl() -> Path:
    df_meta = load_metadata_df()
    pdf_dict = get_PDF_paths()

    all_chunks: list[dict] = []
    for pdf_id in df_meta["id"]:
        pdf_path = pdf_dict.get(pdf_id)
        if not pdf_path:
            logger.warning("pdf_not_found_for_metadata", pdf_id=pdf_id)
            continue

        try:
            chunks = extract_pdf_chunks(pdf_path)
        except Exception as exc:
            logger.warning("extract_chunks_failed", pdf_id=pdf_id, error=str(exc))
            continue

        all_chunks.extend(chunk.model_dump() for chunk in chunks)
        logger.info("chunks_extracted", pdf_id=pdf_id, num_chunks=len(chunks))

    out_path = canonical_chunks_path()
    write_chunks_jsonl(all_chunks, out_path)
    logger.info("chunks_jsonl_built", path=str(out_path), total_chunks=len(all_chunks))
    return out_path


def build_json() -> Path:
    """Backwards-compatible alias; canonical output remains chunks.jsonl."""
    return build_jsonl()


if __name__ == "__main__":
    build_jsonl()
