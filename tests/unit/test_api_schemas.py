from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import IngestRequest, QueryRequest, SearchRequest


def test_search_request_defaults_follow_interface_contract() -> None:
    payload = SearchRequest(question="energy consumption")

    assert payload.top_k == 10
    assert payload.retrieval_mode is None
    assert payload.rerank is False


def test_query_request_defaults_follow_interface_contract() -> None:
    payload = QueryRequest(question="what were the emissions?")

    assert payload.top_k == 5
    assert payload.retrieval_mode is None
    assert payload.rerank is True
    assert payload.llm_backend is None


def test_ingest_request_defaults_follow_interface_contract() -> None:
    payload = IngestRequest()

    assert payload.source == "metadata_csv"
    assert payload.path == ""


def test_ingest_request_allows_empty_path_for_local_dir() -> None:
    payload = IngestRequest(source="local_dir", path="")

    assert payload.source == "local_dir"
    assert payload.path == ""


def test_ingest_request_requires_path_for_url_list() -> None:
    with pytest.raises(ValidationError):
        IngestRequest(source="url_list", path="   ")


def test_search_request_rejects_unknown_retrieval_mode() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(question="test", retrieval_mode="keyword")


def test_query_request_rejects_unknown_llm_backend() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="test", llm_backend="openai")


def test_ingest_request_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        IngestRequest(source="s3")
