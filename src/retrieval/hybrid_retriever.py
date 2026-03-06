from __future__ import annotations

from typing import Any

from src.core.constants import RRF_K
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()

    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        dense_results = self.dense_retriever.retrieve(question, top_k)
        sparse_results = self.sparse_retriever.retrieve(question, top_k)

        fused: dict[str, dict[str, Any]] = {}
        for result_set in (dense_results, sparse_results):
            for item in result_set:
                chunk_id = item["chunk_id"]
                existing = fused.get(chunk_id)
                rrf_score = 1.0 / (RRF_K + int(item["rank"]))
                if existing is None:
                    fused[chunk_id] = {**item, "score": rrf_score}
                else:
                    existing["score"] += rrf_score

        ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        for idx, item in enumerate(ordered, start=1):
            item["rank"] = idx
        return ordered
