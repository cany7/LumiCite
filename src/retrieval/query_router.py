from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    query_type: str
    retrieval_mode: str
    rerank: bool


class QueryRouter:
    def route(self, query: str) -> RetrievalConfig:
        normalized = query.lower().strip()

        if self._is_visual(normalized):
            return RetrievalConfig(query_type="visual", retrieval_mode="hybrid", rerank=True)
        if self._is_comparison(normalized):
            return RetrievalConfig(query_type="comparison", retrieval_mode="hybrid", rerank=True)
        if self._is_boolean(normalized):
            return RetrievalConfig(query_type="boolean", retrieval_mode="sparse", rerank=False)
        if self._is_numeric(normalized):
            return RetrievalConfig(query_type="numeric", retrieval_mode="hybrid", rerank=True)
        return RetrievalConfig(query_type="general", retrieval_mode="dense", rerank=False)

    def _is_numeric(self, query: str) -> bool:
        return any(token in query for token in ("how much", "how many", "what amount", "what percent"))

    def _is_boolean(self, query: str) -> bool:
        return query.startswith(("is ", "are ", "does ", "do ", "did ", "was ", "were ", "can "))

    def _is_comparison(self, query: str) -> bool:
        return any(token in query for token in ("compare", "comparison", "versus", "vs", "difference"))

    def _is_visual(self, query: str) -> bool:
        return any(
            token in query
            for token in ("figure", "table", "chart", "graph", "plot", "diagram", "visual")
        )
