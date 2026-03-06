from __future__ import annotations

from typing import Any

from src.retrieval.dense_retriever import DenseRetriever


def retrieve(question: str, num_chunks: int = 3) -> dict[int, dict[str, Any]]:
    retriever = DenseRetriever()
    results = retriever.retrieve(question, num_chunks)
    payload: dict[int, dict[str, Any]] = {}
    for item in results:
        payload[item["rank"]] = {
            "chunk": item["text"],
            "paper": item["ref_id"],
            "rank": item["rank"],
            "score": item["score"],
            "page": item["page"],
            "source_file": item["source_file"],
            "headings": item["headings"],
        }
    return payload


def get_chunks(question: str, num_chunks: int = 3) -> dict[int, dict[str, Any]]:
    return retrieve(question, num_chunks=num_chunks)
