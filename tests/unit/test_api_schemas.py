from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import QueryRequest, SearchRequest


def test_search_request_defaults_follow_current_contract() -> None:
    payload = SearchRequest(question="energy consumption")

    assert payload.top_k == 10
    assert payload.retrieval_mode is None
    assert payload.rerank is False


def test_query_request_defaults_follow_current_contract() -> None:
    payload = QueryRequest(question="what were the emissions?")

    assert payload.top_k == 5
    assert payload.retrieval_mode is None
    assert payload.rerank is True
    assert payload.llm_backend is None
    assert payload.llm_model is None


def test_query_request_accepts_api_model_override() -> None:
    payload = QueryRequest(question="test", llm_backend="api", llm_model="qwen/custom")

    assert payload.llm_backend == "api"
    assert payload.llm_model == "qwen/custom"


def test_search_request_rejects_unknown_retrieval_mode() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(question="test", retrieval_mode="keyword")


def test_query_request_rejects_unknown_llm_backend() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="test", llm_backend="openai")
