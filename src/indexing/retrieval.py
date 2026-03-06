from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings
from src.core.logging import get_logger, timed
from src.core.paths import find_project_root

logger = get_logger(__name__)


@dataclass
class _RetrievalState:
    model: SentenceTransformer
    index: faiss.Index
    text_data: list[dict[str, Any]]


_STATE: _RetrievalState | None = None


def _paths(root: Path) -> tuple[Path, Path, Path]:
    data_dir = root / "data"
    return (
        data_dir / "my_faiss.index",
        data_dir / "text_data.pkl",
        root / "data" / "JSON" / "embeddings.jsonl",
    )


def _build_state() -> _RetrievalState:
    settings = get_settings()
    root = find_project_root()
    index_path, text_path, embeddings_path = _paths(root)

    model = SentenceTransformer(settings.embedding_model)

    if index_path.exists() and text_path.exists():
        index = faiss.read_index(str(index_path))
        with text_path.open("rb") as handle:
            text_data = pickle.load(handle)
        logger.info("retrieval_index_loaded", index=str(index_path), rows=len(text_data))
        return _RetrievalState(model=model, index=index, text_data=text_data)

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings: list[list[float]] = []
    text_data: list[dict[str, Any]] = []
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            embeddings.append(record["embedding"])
            text_data.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record.get("metadata", {}),
                }
            )

    if not embeddings:
        raise ValueError("No embeddings found to build FAISS index")

    vecs = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with text_path.open("wb") as handle:
        pickle.dump(text_data, handle)

    logger.info("retrieval_index_built", vectors=index.ntotal, dim=vecs.shape[1])
    return _RetrievalState(model=model, index=index, text_data=text_data)


def _state() -> _RetrievalState:
    global _STATE
    if _STATE is None:
        _STATE = _build_state()
    return _STATE


@timed("retrieve")
def retrieve(question: str, num_chunks: int = 3) -> dict[int, dict[str, Any]]:
    settings = get_settings()
    state = _state()

    if num_chunks <= 0:
        return {}

    query_vec = state.model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = state.index.search(query_vec, num_chunks)

    best_dist = float(distances[0][0])
    if best_dist > settings.distance_threshold:
        logger.warning(
            "retrieve_below_threshold",
            best_distance=best_dist,
            threshold=settings.distance_threshold,
        )
        return {}

    results: dict[int, dict[str, Any]] = {}
    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0 or idx >= len(state.text_data):
            continue

        record = state.text_data[idx]
        metadata = record.get("metadata", {}) or {}
        doc_id = (
            metadata.get("doc_id")
            or metadata.get("paper_id")
            or metadata.get("source_id")
            or metadata.get("paper")
            or ""
        )
        if not doc_id:
            rec_id = str(record.get("id", ""))
            doc_id = rec_id.split("_", 1)[0] if "_" in rec_id else rec_id

        distance = float(distances[0][rank - 1])
        score = 1.0 / (1.0 + distance)

        results[rank] = {
            "chunk": record.get("text", ""),
            "paper": doc_id,
            "rank": rank,
            "score": score,
            "page": metadata.get("page_number") or metadata.get("page"),
            "source_file": metadata.get("source_file", ""),
            "headings": metadata.get("headings", []),
        }

    return results


def get_chunks(question: str, num_chunks: int = 3) -> dict[int, dict[str, Any]]:
    return retrieve(question, num_chunks=num_chunks)
