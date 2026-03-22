from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import src.api.schemas as schemas_module
from src.api.schemas import QueryRequest, SearchRequest


def test_search_request_defaults_follow_current_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schemas_module, "get_settings", lambda: SimpleNamespace(retrieval_top_k=7))
    payload = SearchRequest(question="energy consumption")

    assert payload.top_k == 7
    assert payload.retrieval_mode is None
    assert payload.rerank is False
    assert payload.query_explanation is True


def test_query_request_defaults_follow_current_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schemas_module, "get_settings", lambda: SimpleNamespace(retrieval_top_k=9))
    payload = QueryRequest(question="what were the emissions?")

    assert payload.top_k == 9
    assert payload.retrieval_mode is None
    assert payload.rerank is False
    assert payload.query_explanation is True
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
