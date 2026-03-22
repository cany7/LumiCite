from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config.settings import get_settings
from src.core.logging import get_logger, timed
from src.core.model_assets import configure_runtime_cache_environment
from src.core.paths import sentence_transformers_cache_dir
from src.core.schemas import ChunkModel, ChunkType, EmbeddingRecord, FigureChunk, TableChunk, TextChunk

logger = get_logger(__name__)


def chunk_from_payload(payload: dict[str, Any]) -> ChunkModel:
    chunk_type = ChunkType(str(payload.get("chunk_type", "text")))
    if chunk_type == ChunkType.TABLE:
        return TableChunk(**payload)
    if chunk_type == ChunkType.FIGURE:
        return FigureChunk(**payload)
    return TextChunk(**payload)


def load_canonical_chunks_jsonl(path: Path) -> list[ChunkModel]:
    chunks: list[ChunkModel] = []
    if not path.exists():
        return chunks

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(chunk_from_payload(json.loads(line)))
    return chunks


def build_embedding_inputs(chunks: list[ChunkModel]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for chunk in chunks:
        caption = getattr(chunk, "caption", "")
        asset_path = getattr(chunk, "asset_path", "")
        inputs.append(
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "page_number": chunk.page_number,
                    "headings": list(chunk.headings),
                    "chunk_type": chunk.chunk_type.value,
                    "caption": caption,
                    "asset_path": asset_path,
                },
            }
        )
    return inputs


def embed_local(records: list[dict[str, Any]], model_name: str, batch_size: int) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    configure_runtime_cache_environment()
    model = SentenceTransformer(model_name, cache_folder=str(sentence_transformers_cache_dir()))
    texts = [record["text"] for record in records]

    vectors: list[list[float]] = []
    for offset in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[offset : offset + batch_size]
        embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        vectors.extend(embeddings.tolist())
    return vectors


@timed("embed_chunks")
def embed_chunks(records: list[dict[str, Any]], model_name: str, batch_size: int) -> list[list[float]]:
    return embed_local(records, model_name=model_name, batch_size=batch_size)


def build_embedding_records(
    chunks: list[ChunkModel],
    *,
    embedding_model: str | None = None,
    batch_size: int | None = None,
) -> list[EmbeddingRecord]:
    settings = get_settings()
    model_name = embedding_model or settings.embedding_model
    effective_batch_size = batch_size or settings.embedding_batch_size
    inputs = build_embedding_inputs(chunks)
    if not inputs:
        return []

    vectors = embed_chunks(inputs, model_name=model_name, batch_size=effective_batch_size)
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    records: list[EmbeddingRecord] = []
    for payload, vector in zip(inputs, vectors):
        text = payload["text"]
        records.append(
            EmbeddingRecord(
                id=payload["id"],
                text=text,
                metadata=payload["metadata"],
                embedding=vector,
                content_hash=hashlib.md5(text.encode("utf-8")).hexdigest(),
                embedding_model=model_name,
                created_at=created_at,
            )
        )
    return records


def write_embeddings_jsonl(records: list[EmbeddingRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
