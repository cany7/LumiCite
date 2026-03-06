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
            ref_id = str(record.get("doc_id", ""))
            if not ref_id:
                chunk_id = str(record.get("chunk_id", ""))
                ref_id = chunk_id.split("_", 1)[0] if "_" in chunk_id else chunk_id

            results.append(
                {
                    "rank": len(results) + 1,
                    "chunk_id": str(record.get("chunk_id", "")),
                    "ref_id": ref_id,
                    "score": score,
                    "text": str(record.get("text", "")),
                    "page": record.get("page_number"),
                    "source_file": str(record.get("source_file", "")),
                    "headings": list(record.get("headings", []) or []),
                    "chunk_type": str(record.get("chunk_type", "text")),
                }
            )

        return results
