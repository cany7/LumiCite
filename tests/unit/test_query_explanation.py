from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.retrieval.query_explanation as query_explanation_module
from src.core.errors import GenerationError
from src.core.constants import RRF_K


class FakeRetriever:
    def __init__(self, responses: dict[str, list[dict]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        self.calls.append((question, top_k))
        return [dict(item) for item in self.responses.get(question, [])[:top_k]]


def test_retrieve_with_query_explanation_merges_candidates_then_reranks(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = FakeRetriever(
        {
            "original question": [
                {"rank": 1, "chunk_id": "chunk-a", "doc_id": "paper1", "score": 0.9, "text": "alpha baseline"},
                {"rank": 2, "chunk_id": "chunk-b", "doc_id": "paper2", "score": 0.8, "text": "beta metric"},
            ],
            "expanded retrieval query": [
                {"rank": 1, "chunk_id": "chunk-b", "doc_id": "paper2", "score": 0.95, "text": "beta metric"},
                {"rank": 2, "chunk_id": "chunk-c", "doc_id": "paper3", "score": 0.85, "text": "gamma denominator"},
            ],
        }
    )
    captured: dict[str, object] = {}

    class FakeClient:
        def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt
            return "expanded retrieval query"

    class FakeReranker:
        def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
            captured["rerank_query"] = query
            captured["candidate_ids"] = [item["chunk_id"] for item in candidates]
            return [{**item, "rank": index} for index, item in enumerate(candidates[:top_k], start=1)]

    monkeypatch.setattr(query_explanation_module, "get_retriever", lambda retrieval_mode: retriever)
    monkeypatch.setattr(query_explanation_module, "create_llm_client", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(query_explanation_module, "Reranker", lambda: FakeReranker())
    monkeypatch.setattr(
        query_explanation_module,
        "get_settings",
        lambda: SimpleNamespace(
            api_model="qwen/default",
            api_key="test-api-key",
            api_base_url="https://api.example.com/v1",
            request_retry_attempts=1,
            request_retry_delay_seconds=0.0,
        ),
    )

    execution = query_explanation_module.retrieve_with_optional_query_explanation(
        "original question",
        top_k=2,
        retrieval_mode="hybrid",
        rerank=True,
        query_explanation=query_explanation_module.QueryExplanationConfig(enabled=True),
    )

    assert execution.expanded_query == "expanded retrieval query"
    assert retriever.calls == [("original question", 10), ("expanded retrieval query", 10)]
    assert captured["rerank_query"] == "expanded retrieval query"
    assert set(captured["candidate_ids"]) == {"chunk-a", "chunk-b", "chunk-c"}
    assert execution.results[0]["rank"] == 1
    assert "User question:\noriginal question" in str(captured["prompt"])
    assert captured["system_prompt"] == query_explanation_module.QUERY_EXPLANATION_SYSTEM_PROMPT


def test_retrieve_without_query_explanation_reranks_with_original_question(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = FakeRetriever(
        {
            "original question": [
                {"rank": 1, "chunk_id": "chunk-a", "doc_id": "paper1", "score": 0.9, "text": "alpha baseline"},
                {"rank": 2, "chunk_id": "chunk-b", "doc_id": "paper2", "score": 0.8, "text": "beta metric"},
            ],
        }
    )
    captured: dict[str, object] = {}

    class FakeReranker:
        def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
            captured["rerank_query"] = query
            return [{**item, "rank": index} for index, item in enumerate(candidates[:top_k], start=1)]

    monkeypatch.setattr(query_explanation_module, "get_retriever", lambda retrieval_mode: retriever)
    monkeypatch.setattr(query_explanation_module, "Reranker", lambda: FakeReranker())

    execution = query_explanation_module.retrieve_with_optional_query_explanation(
        "original question",
        top_k=1,
        retrieval_mode="hybrid",
        rerank=True,
        query_explanation=query_explanation_module.QueryExplanationConfig(enabled=False),
    )

    assert retriever.calls == [("original question", 5)]
    assert captured["rerank_query"] == "original question"
    assert execution.expanded_query is None
    assert [item["chunk_id"] for item in execution.results] == ["chunk-a"]


def test_retrieve_with_query_explanation_falls_back_to_original_query(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = FakeRetriever(
        {
            "original question": [
                {"rank": 1, "chunk_id": "chunk-a", "doc_id": "paper1", "score": 0.9, "text": "alpha baseline"},
                {"rank": 2, "chunk_id": "chunk-b", "doc_id": "paper2", "score": 0.8, "text": "beta metric"},
            ],
        }
    )

    class FailingClient:
        def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:  # noqa: ARG002
            raise GenerationError(
                error_type="generation_request_error",
                message="request failed",
                retryable=False,
            )

    monkeypatch.setattr(query_explanation_module, "get_retriever", lambda retrieval_mode: retriever)
    monkeypatch.setattr(query_explanation_module, "create_llm_client", lambda *args, **kwargs: FailingClient())
    monkeypatch.setattr(
        query_explanation_module,
        "get_settings",
        lambda: SimpleNamespace(
            api_model="qwen/default",
            api_key="test-api-key",
            api_base_url="https://api.example.com/v1",
            request_retry_attempts=1,
            request_retry_delay_seconds=0.0,
        ),
    )

    execution = query_explanation_module.retrieve_with_optional_query_explanation(
        "original question",
        top_k=1,
        retrieval_mode="hybrid",
        rerank=False,
        query_explanation=query_explanation_module.QueryExplanationConfig(enabled=True),
    )

    assert execution.expanded_query is None
    assert retriever.calls == [("original question", 1)]
    assert [item["chunk_id"] for item in execution.results] == ["chunk-a"]


def test_fuse_ranked_result_sets_uses_rrf() -> None:
    results = query_explanation_module._fuse_ranked_result_sets(
        [
            [
                {"rank": 1, "chunk_id": "chunk-a", "score": 0.9},
                {"rank": 2, "chunk_id": "chunk-b", "score": 0.8},
            ],
            [
                {"rank": 1, "chunk_id": "chunk-a", "score": 0.7},
                {"rank": 2, "chunk_id": "chunk-c", "score": 0.6},
            ],
        ],
        top_k=3,
    )

    assert results[0]["chunk_id"] == "chunk-a"
    assert results[0]["score"] == pytest.approx((1 / (RRF_K + 1)) + (1 / (RRF_K + 1)))
