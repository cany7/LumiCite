from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config.settings import get_settings
from src.core.logging import get_logger, timed
from src.core.paths import find_project_root
from src.core.schemas import ChunkType, EmbeddingRecord, FigureChunk, TableChunk, TextChunk
from src.ingestion.legacy_loader import load_legacy_chunks_json

logger = get_logger(__name__)

ChunkModel = TextChunk | TableChunk | FigureChunk


def _model_from_payload(payload: dict[str, Any]) -> ChunkModel:
    chunk_type = ChunkType(str(payload.get("chunk_type", "text")))
    if chunk_type == ChunkType.TABLE:
        return TableChunk(**payload)
    if chunk_type == ChunkType.FIGURE:
        return FigureChunk(**payload)
    return TextChunk(**payload)


def load_canonical_chunks_jsonl(path: Path) -> list[ChunkModel]:
    chunks: list[ChunkModel] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chunks.append(_model_from_payload(payload))
    return chunks


def load_ingestion_chunks(json_dir: Path) -> list[ChunkModel]:
    canonical_path = json_dir / "chunks.jsonl"
    legacy_chunks_path = json_dir / "chunks.json"
    legacy_alt_path = json_dir / "alt_text.json"

    if canonical_path.exists():
        logger.info("loading_canonical_chunks", path=str(canonical_path))
        return load_canonical_chunks_jsonl(canonical_path)

    chunks: list[ChunkModel] = []
    if legacy_chunks_path.exists():
        chunks.extend(load_legacy_chunks_json(legacy_chunks_path))
    if legacy_alt_path.exists():
        chunks.extend(load_legacy_chunks_json(legacy_alt_path))

    if not chunks:
        raise FileNotFoundError("No chunks found in canonical JSONL or legacy JSON files.")

    logger.info("loading_legacy_chunks", count=len(chunks))
    return chunks


def embed_local(records: list[dict[str, Any]], model_name: str, batch_size: int) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [record["text"] for record in records]

    vectors: list[list[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i : i + batch_size]
        embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        vectors.extend(embs.tolist())
    return vectors


@timed("embed_chunks")
def embed_chunks(records: list[dict[str, Any]], model_name: str, batch_size: int) -> list[list[float]]:
    return embed_local(records, model_name=model_name, batch_size=batch_size)


def write_embeddings_jsonl(records: list[EmbeddingRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def main() -> None:
    settings = get_settings()
    root = find_project_root()
    json_dir = root / "data" / "JSON"
    json_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_ingestion_chunks(json_dir)

    records_for_embedding: list[dict[str, Any]] = []
    for chunk in chunks:
        records_for_embedding.append(
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "page_number": chunk.page_number,
                    "headings": chunk.headings,
                    "source_file": chunk.source_file,
                    "chunk_type": chunk.chunk_type.value,
                },
            }
        )

    vectors = embed_chunks(
        records_for_embedding,
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    embedded_records: list[EmbeddingRecord] = []
    for record, vector in zip(records_for_embedding, vectors):
        text = record["text"]
        embedded_records.append(
            EmbeddingRecord(
                id=record["id"],
                text=text,
                metadata=record["metadata"],
                embedding=vector,
                content_hash=hashlib.md5(text.encode("utf-8")).hexdigest(),
                embedding_model=settings.embedding_model,
                created_at=now,
            )
        )

    out_path = json_dir / "embeddings.jsonl"
    write_embeddings_jsonl(embedded_records, out_path)
    logger.info("embeddings_written", path=str(out_path), num_records=len(embedded_records))


if __name__ == "__main__":
    main()
