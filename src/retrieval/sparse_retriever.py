from __future__ import annotations

from typing import Any

from src.core.logging import timed
from src.indexing.bm25_index import BM25Index


class SparseRetriever:
    def __init__(self, index: BM25Index | None = None) -> None:
        self.index = index or BM25Index()

    @timed("retrieve")
    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        rows = self.index.search(question, top_k)
        results: list[dict[str, Any]] = []
        for score, record in rows:
            doc_id = str(record.get("doc_id", ""))
            if not doc_id:
                chunk_id = str(record.get("chunk_id", ""))
                doc_id = chunk_id.split("_", 1)[0] if "_" in chunk_id else chunk_id

            results.append(
                {
                    "rank": len(results) + 1,
                    "doc_id": doc_id,
                    "chunk_id": str(record.get("chunk_id", "")),
                    "chunk_type": str(record.get("chunk_type", "text") or "text"),
                    "score": score,
                    "text": str(record.get("text", "")),
                    "page_number": record.get("page_number"),
                    "headings": list(record.get("headings", []) or []),
                    "caption": str(record.get("caption", "")),
                    "asset_path": str(record.get("asset_path", "")),
                }
            )

        return results
