from __future__ import annotations

from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from src.config.settings import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _cross_encoder(model_name: str) -> CrossEncoder:
    logger.info("reranker_model_loading", model=model_name)
    return CrossEncoder(model_name)


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not candidates:
            return []

        pairs = [(query, str(candidate.get("text", ""))) for candidate in candidates]
        scores = _cross_encoder(self.model_name).predict(pairs)
        rescored: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            rescored.append({**candidate, "score": float(score)})

        rescored.sort(key=lambda item: item["score"], reverse=True)
        rescored = rescored[:top_k]
        for idx, item in enumerate(rescored, start=1):
            item["rank"] = idx
        return rescored
