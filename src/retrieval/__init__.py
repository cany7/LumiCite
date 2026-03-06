from __future__ import annotations

from src.config.settings import get_settings
from src.retrieval.dense_retriever import DenseRetriever


def get_retriever(retrieval_mode: str | None = None):
    settings = get_settings()
    mode = retrieval_mode or settings.retrieval_mode
    if mode == "dense":
        return DenseRetriever()

    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.sparse_retriever import SparseRetriever

    if mode == "sparse":
        return SparseRetriever()
    if mode == "hybrid":
        return HybridRetriever()
    raise ValueError("retrieval_mode must be one of: dense, sparse, hybrid")
