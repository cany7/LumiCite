from __future__ import annotations

from functools import lru_cache
from typing import Any
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings
from src.core.logging import get_logger, timed
from src.indexing.vector_store import FaissStore

logger = get_logger(__name__)


def _doc_id_from_record(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {}) or {}
    doc_id = (
        metadata.get("doc_id")
        or metadata.get("paper_id")
        or metadata.get("source_id")
        or metadata.get("paper")
        or ""
    )
    if doc_id:
        return str(doc_id)

    record_id = str(record.get("id", ""))
    return record_id.split("_", 1)[0] if "_" in record_id else record_id


@lru_cache(maxsize=1)
def _embedding_model(model_name: str) -> SentenceTransformer:
    logger.info("dense_model_loading", model=model_name)
    return SentenceTransformer(model_name)


class DenseRetriever:
    def __init__(
        self,
        store: FaissStore | None = None,
        model_name: str | None = None,
        distance_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.distance_threshold = (
            settings.distance_threshold if distance_threshold is None else distance_threshold
        )
        self.store = store or FaissStore()

    def _model(self) -> SentenceTransformer:
        return _embedding_model(self.model_name)

    @timed("retrieve")
    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        self.store.ensure_loaded()
        query_vec = self._model().encode([question], convert_to_numpy=True).astype("float32")
        distances, indices = self.store.search(query_vec, top_k)

        if len(distances[0]) == 0:
            return []

        best_distance = float(distances[0][0])
        if best_distance > self.distance_threshold:
            logger.warning(
                "dense_retrieve_below_threshold",
                best_distance=best_distance,
                threshold=self.distance_threshold,
            )
            return []

        results: list[dict[str, Any]] = []
        for offset, row_idx in enumerate(indices[0]):
            if row_idx < 0 or row_idx >= len(self.store.text_data):
                continue

            record = self.store.text_data[row_idx]
            metadata = record.get("metadata", {}) or {}
            distance = float(distances[0][offset])
            score = 1.0 / (1.0 + distance)
            results.append(
                {
                    "rank": len(results) + 1,
                    "chunk_id": str(record.get("id", "")),
                    "ref_id": _doc_id_from_record(record),
                    "score": score,
                    "text": str(record.get("text", "")),
                    "page": metadata.get("page_number") or metadata.get("page"),
                    "source_file": str(metadata.get("source_file", "")),
                    "headings": list(metadata.get("headings", []) or []),
                    "chunk_type": str(metadata.get("chunk_type", "text")),
                }
            )

        return results
