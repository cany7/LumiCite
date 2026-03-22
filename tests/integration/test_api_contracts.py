from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import src.api.routes as routes_module
from src.api.app import create_app
from src.core.schemas import Citation, ChunkType, RAGAnswer, VerificationResult


def _write_chunks(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "chunk_id": "paper1_deadbeef",
            "doc_id": "paper1",
            "text": "alpha evidence",
            "chunk_type": "text",
            "page_number": 1,
            "headings": ["Intro"],
        },
        {
            "chunk_id": "paper2_fig_cafebabe",
            "doc_id": "paper2",
            "text": "figure summary\nCaption: Figure 2. Trend",
            "chunk_type": "figure",
            "page_number": 2,
            "headings": ["Results"],
            "caption": "Figure 2. Trend",
            "asset_path": "data/assets/paper2/paper2_fig_cafebabe.png",
        },
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_api_routes_follow_current_phase4_contracts(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    _write_chunks(root / "data" / "metadata" / "chunks" / "chunks.jsonl")

    class FakePipeline:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config

        def answer_question(self, question: str, **kwargs) -> RAGAnswer:  # noqa: ANN003
            return RAGAnswer(
                answer="552 tCO2e",
                supporting_materials="alpha evidence",
                explanation="grounded answer",
                citations=[
                    Citation(
                        doc_id="paper2",
                        chunk_id="paper2_fig_cafebabe",
                        page_number=2,
                        evidence_text="figure summary",
                        evidence_type=ChunkType.FIGURE,
                        headings=["Results"],
                        caption="Figure 2. Trend",
                        asset_path="data/assets/paper2/paper2_fig_cafebabe.png",
                    )
                ],
                retrieval_latency_ms=12.3,
                generation_latency_ms=45.6,
                retrieval_mode="hybrid",
                llm_backend="api",
                verification=VerificationResult(passed=True, confidence=0.95, warnings=[]),
            )

    monkeypatch.setattr(routes_module, "find_project_root", lambda: root)
    monkeypatch.setattr(
        routes_module,
        "get_settings",
        lambda: SimpleNamespace(
            embedding_model="mock-model",
            retrieval_mode="hybrid",
            llm_backend="api",
            api_model="qwen/default",
            api_key="test-api-key",
            api_base_url="https://api.example.com/v1",
            query_explanation_reasoning_effort="none",
        ),
    )
    monkeypatch.setattr(routes_module, "_index_loaded", lambda: True)
    monkeypatch.setattr(
        routes_module,
        "_retrieve_results",
        lambda question, top_k, retrieval_mode, rerank, query_explanation=None: [
            {
                "rank": 1,
                "doc_id": "paper2",
                "chunk_id": "paper2_fig_cafebabe",
                "chunk_type": "figure",
                "score": 0.91,
                "text": "figure summary\nCaption: Figure 2. Trend",
                "page_number": 2,
                "headings": ["Results"],
                "caption": "Figure 2. Trend",
                "asset_path": "data/assets/paper2/paper2_fig_cafebabe.png",
            }
        ],
    )
    monkeypatch.setattr(routes_module, "_build_rag_pipeline", lambda config: FakePipeline(config))

    client = TestClient(create_app())

    health = client.get("/api/v1/health")
    papers = client.get("/api/v1/papers")
    search = client.post("/api/v1/search", json={"question": "What does Figure 2 show?"})
    query = client.post("/api/v1/query", json={"question": "What does Figure 2 show?"})
    missing = client.post("/api/v1/ingest")

    assert health.status_code == 200
    assert health.json()["num_papers"] == 2
    assert health.json()["num_chunks"] == 2
    assert health.json()["retrieval_modes_available"] == ["dense", "sparse", "hybrid"]

    assert papers.status_code == 200
    assert papers.json() == {
        "papers": [
            {"doc_id": "paper1", "num_chunks": 1},
            {"doc_id": "paper2", "num_chunks": 1},
        ],
        "total": 2,
    }

    search_payload = search.json()
    assert search.status_code == 200
    assert search_payload["retrieval_mode"] == "hybrid"
    assert search_payload["results"][0]["chunk_type"] == "figure"
    assert search_payload["results"][0]["caption"] == "Figure 2. Trend"
    assert search_payload["results"][0]["asset_path"].endswith(".png")

    query_payload = query.json()
    assert query.status_code == 200
    assert query_payload["llm_backend"] == "api"
    assert "answer_value" not in query_payload
    assert "answer_unit" not in query_payload
    assert query_payload["citations"][0]["doc_id"] == "paper2"
    assert query_payload["citations"][0]["caption"] == "Figure 2. Trend"
    assert query_payload["citations"][0]["asset_path"].endswith(".png")

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "VALIDATION_ERROR"


def test_api_routes_accept_query_explanation_flags(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    _write_chunks(root / "data" / "metadata" / "chunks" / "chunks.jsonl")
    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, config) -> None:  # noqa: ANN001
            captured["config"] = config

        def answer_question(self, question: str, **kwargs) -> RAGAnswer:  # noqa: ANN003
            captured["question_kwargs"] = kwargs
            return RAGAnswer(
                answer="552 tCO2e",
                supporting_materials="alpha evidence",
                explanation="grounded answer",
                citations=[],
                retrieval_latency_ms=12.3,
                generation_latency_ms=45.6,
                retrieval_mode="hybrid",
                llm_backend="api",
                verification=VerificationResult(passed=True, confidence=0.95, warnings=[]),
            )

    def fake_retrieve_results(question, top_k, retrieval_mode, rerank, query_explanation=None):  # noqa: ANN001, ANN201
        captured["search_query_explanation"] = query_explanation
        return [
            {
                "rank": 1,
                "doc_id": "paper1",
                "chunk_id": "paper1_deadbeef",
                "chunk_type": "text",
                "score": 0.91,
                "text": "alpha evidence",
                "page_number": 1,
                "headings": ["Intro"],
                "caption": "",
                "asset_path": "",
            }
        ]

    monkeypatch.setattr(routes_module, "find_project_root", lambda: root)
    monkeypatch.setattr(
        routes_module,
        "get_settings",
        lambda: SimpleNamespace(
            embedding_model="mock-model",
            retrieval_mode="hybrid",
            llm_backend="api",
            api_model="qwen/default",
            api_key="test-api-key",
            api_base_url="https://api.example.com/v1",
            query_explanation_reasoning_effort="minimal",
        ),
    )
    monkeypatch.setattr(routes_module, "_index_loaded", lambda: True)
    monkeypatch.setattr(routes_module, "_retrieve_results", fake_retrieve_results)
    monkeypatch.setattr(routes_module, "_build_rag_pipeline", lambda config: FakePipeline(config))

    client = TestClient(create_app())

    search = client.post("/api/v1/search", json={"question": "trend", "query_explanation": True})
    query = client.post("/api/v1/query", json={"question": "trend", "query_explanation": True})

    assert search.status_code == 200
    assert captured["search_query_explanation"].enabled is True
    assert captured["search_query_explanation"].llm_model == "qwen/default"
    assert captured["search_query_explanation"].reasoning_effort == "minimal"
    assert captured["config"].query_explanation_enabled is True
    assert captured["question_kwargs"]["query_explanation_enabled"] is True
    assert query.status_code == 200
