from __future__ import annotations

import pytest

from src.retrieval.query_router import QueryRouter, RetrievalConfig


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What does Figure 3 show?", RetrievalConfig("visual", "hybrid", True)),
        ("Compare GPT-3 versus PaLM emissions", RetrievalConfig("comparison", "hybrid", True)),
        ("Is the model carbon neutral?", RetrievalConfig("boolean", "sparse", False)),
        ("How much CO2 was emitted?", RetrievalConfig("numeric", "hybrid", True)),
        ("Explain the training setup", RetrievalConfig("general", "dense", False)),
    ],
)
def test_query_router_returns_stable_defaults(query: str, expected: RetrievalConfig) -> None:
    assert QueryRouter().route(query) == expected

