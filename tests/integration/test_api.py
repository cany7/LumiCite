from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api.routes as routes_module
import src.ingestion.sources as sources_module
from src.api.app import create_app
from src.config.settings import get_settings
from src.core.schemas import Citation, RAGAnswer, VerificationResult
from src.ingestion.sources.base import DocumentMeta


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_metadata(root: Path, rows: list[dict[str, object]]) -> None:
    metadata_path = root / "data" / "metadata" / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "year"])
        writer.writeheader()
        writer.writerows(rows)


def _write_chunks(root: Path, rows: list[dict[str, object]]) -> None:
    chunks_path = root / "data" / "JSON" / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def api_root(tmp_path: Path) -> Path:
    _write_metadata(
        tmp_path,
        [
            {"id": "paper1", "title": "Paper One", "year": 2021},
            {"id": "paper2", "title": "Paper Two", "year": 2022},
        ],
    )
    _write_chunks(
        tmp_path,
        [
            {
                "chunk_id": "paper1_deadbeef",
                "doc_id": "paper1",
                "text": "alpha evidence",
                "page_number": 3,
                "source_file": "paper1.pdf",
                "headings": ["Results"],
            },
            {
                "chunk_id": "paper1_feedface",
                "doc_id": "paper1",
                "text": "beta evidence",
                "page_number": 4,
                "source_file": "paper1.pdf",
                "headings": ["Discussion"],
            },
            {
                "chunk_id": "paper2_deadbeef",
                "doc_id": "paper2",
                "text": "gamma evidence",
                "page_number": 7,
                "source_file": "paper2.pdf",
                "headings": ["Abstract"],
            },
        ],
    )
    return tmp_path


def _make_client(monkeypatch: pytest.MonkeyPatch, root: Path, *, index_loaded: bool = True) -> TestClient:
    monkeypatch.setattr(routes_module, "find_project_root", lambda: root)
    monkeypatch.setattr(routes_module, "_index_loaded", lambda: index_loaded)
    return TestClient(create_app())


def test_health_endpoint_reports_contract_and_corpus_stats(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "tests/minilm")
    client = _make_client(monkeypatch, api_root, index_loaded=True)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "status",
        "index_loaded",
        "num_papers",
        "num_chunks",
        "embedding_model",
        "retrieval_modes_available",
    }
    assert payload["status"] == "ok"
    assert payload["index_loaded"] is True
    assert payload["num_papers"] == 2
    assert payload["num_chunks"] == 3
    assert payload["embedding_model"] == "tests/minilm"
    assert payload["retrieval_modes_available"] == ["dense", "sparse", "hybrid"]


def test_health_num_papers_matches_papers_total(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    client = _make_client(monkeypatch, api_root)

    health_response = client.get("/api/v1/health")
    papers_response = client.get("/api/v1/papers")

    assert health_response.status_code == 200
    assert papers_response.status_code == 200
    assert health_response.json()["num_papers"] == papers_response.json()["total"]


def test_papers_endpoint_uses_metadata_titles_and_chunk_counts(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    client = _make_client(monkeypatch, api_root)

    response = client.get("/api/v1/papers")

    assert response.status_code == 200
    payload = response.json()
    papers = {paper["id"]: paper for paper in payload["papers"]}
    assert payload["total"] == 2
    assert papers["paper1"] == {
        "id": "paper1",
        "title": "Paper One",
        "year": 2021,
        "num_chunks": 2,
    }
    assert papers["paper2"] == {
        "id": "paper2",
        "title": "Paper Two",
        "year": 2022,
        "num_chunks": 1,
    }


def test_papers_endpoint_includes_metadata_only_and_chunk_only_documents(
    monkeypatch: pytest.MonkeyPatch,
    api_root: Path,
) -> None:
    _write_metadata(
        api_root,
        [
            {"id": "paper1", "title": "Paper One", "year": 2021},
            {"id": "paper2", "title": "Paper Two", "year": 2022},
            {"id": "paper3", "title": "Paper Three", "year": 2023},
        ],
    )
    _write_chunks(
        api_root,
        [
            {
                "chunk_id": "paper1_deadbeef",
                "doc_id": "paper1",
                "text": "alpha evidence",
                "page_number": 3,
                "source_file": "paper1.pdf",
                "headings": ["Results"],
            },
            {
                "chunk_id": "orphan_deadbeef",
                "doc_id": "orphan",
                "text": "orphan evidence",
                "page_number": 9,
                "source_file": "orphan.pdf",
                "headings": ["Appendix"],
            },
        ],
    )
    client = _make_client(monkeypatch, api_root)

    response = client.get("/api/v1/papers")

    assert response.status_code == 200
    payload = response.json()
    papers = {paper["id"]: paper for paper in payload["papers"]}
    assert payload["total"] == 4
    assert papers["paper3"] == {
        "id": "paper3",
        "title": "Paper Three",
        "year": 2023,
        "num_chunks": 0,
    }
    assert papers["orphan"] == {
        "id": "orphan",
        "title": "orphan",
        "year": 0,
        "num_chunks": 1,
    }


def test_search_endpoint_uses_settings_default_mode_and_shape(
    monkeypatch: pytest.MonkeyPatch,
    api_root: Path,
) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "sparse")
    captured: dict[str, object] = {}

    def fake_retrieve(question: str, top_k: int, retrieval_mode: str, rerank: bool) -> list[dict[str, object]]:
        captured.update(
            {
                "question": question,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "rerank": rerank,
            }
        )
        return [
            {
                "rank": 1,
                "ref_id": "paper1",
                "score": 0.91,
                "text": "alpha evidence",
                "page": 3,
                "source_file": "paper1.pdf",
                "headings": ["Results"],
            }
        ]

    monkeypatch.setattr(routes_module, "_retrieve_results", fake_retrieve)
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/search", json={"question": "energy question"})

    assert response.status_code == 200
    assert captured == {
        "question": "energy question",
        "top_k": 10,
        "retrieval_mode": "sparse",
        "rerank": False,
    }
    payload = response.json()
    assert payload["retrieval_mode"] == "sparse"
    assert payload["total_results"] == 1
    assert isinstance(payload["retrieval_latency_ms"], float)
    assert payload["results"] == [
        {
            "rank": 1,
            "ref_id": "paper1",
            "score": 0.91,
            "text": "alpha evidence",
            "page": 3,
            "source_file": "paper1.pdf",
            "headings": ["Results"],
        }
    ]


def test_search_index_not_loaded_uses_error_schema(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    monkeypatch.setattr(routes_module, "_retrieve_results", lambda *args, **kwargs: (_ for _ in ()).throw(
        FileNotFoundError("chunks file not found")
    ))
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/search", json={"question": "energy question"})

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "INDEX_NOT_LOADED"
    assert payload["error"]["message"]
    assert "chunks file not found" in payload["error"]["detail"]


def test_query_endpoint_uses_settings_defaults_and_returns_rag_answer(
    monkeypatch: pytest.MonkeyPatch,
    api_root: Path,
) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "dense")
    monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
    captured: dict[str, object] = {}

    class StubPipeline:
        def __init__(self, config=None) -> None:  # noqa: ANN001
            captured["config"] = config

        def answer_question(self, question: str, **kwargs: object) -> RAGAnswer:
            captured["question"] = question
            captured["kwargs"] = kwargs
            return RAGAnswer(
                answer="552 tCO2e",
                answer_value="552",
                answer_unit="tCO2e",
                ref_id=["patterson2021"],
                supporting_materials="Training GPT-3 resulted in 552 tCO2e.",
                explanation="The cited evidence states the total emissions.",
                citations=[
                    Citation(
                        ref_id="patterson2021",
                        page=8,
                        evidence_text="Training GPT-3 resulted in 552 tCO2e.",
                        evidence_type="text",
                    )
                ],
                retrieval_latency_ms=12.5,
                generation_latency_ms=34.2,
                retrieval_mode=str(kwargs["retrieval_mode"]),
                llm_backend=str(kwargs["llm_backend"]),
                verification=VerificationResult(
                    passed=True,
                    confidence=0.9,
                    warnings=[],
                    corrected_output=None,
                ),
            )

    monkeypatch.setattr(routes_module, "_build_rag_pipeline", lambda config: StubPipeline(config=config))
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/query", json={"question": "test"})

    assert response.status_code == 200
    assert captured["question"] == "test"
    assert captured["kwargs"] == {
        "top_k": 5,
        "retrieval_mode": "dense",
        "rerank": True,
        "llm_backend": "ollama",
    }
    config = captured["config"]
    assert getattr(config, "top_k") == 5
    assert getattr(config, "retrieval_mode") == "dense"
    assert getattr(config, "rerank") is True
    assert getattr(config, "llm_backend") == "ollama"

    payload = response.json()
    assert payload["answer"] == "552 tCO2e"
    assert payload["retrieval_mode"] == "dense"
    assert payload["llm_backend"] == "ollama"
    assert payload["citations"][0]["page"] == 8
    assert payload["verification"]["passed"] is True


def test_query_invalid_body_uses_validation_error_schema(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/query", json={})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["message"]
    assert payload["error"]["detail"]


def test_query_index_not_loaded_uses_error_schema(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    class FailingPipeline:
        def answer_question(self, question: str, **kwargs: object) -> RAGAnswer:  # pragma: no cover - unreachable
            raise FileNotFoundError("embeddings file not found")

    monkeypatch.setattr(routes_module, "_build_rag_pipeline", lambda config: FailingPipeline())
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/query", json={"question": "test"})

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "INDEX_NOT_LOADED"
    assert payload["error"]["message"]
    assert "embeddings file not found" in payload["error"]["detail"]


def test_query_generation_failures_use_error_schema(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    class FailingPipeline:
        def answer_question(self, question: str, **kwargs: object) -> RAGAnswer:  # pragma: no cover - unreachable
            raise RuntimeError("backend boom")

    monkeypatch.setattr(routes_module, "_build_rag_pipeline", lambda config: FailingPipeline())
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/query", json={"question": "test"})

    assert response.status_code == 500
    payload = response.json()
    assert payload["error"]["code"] == "GENERATION_FAILED"
    assert payload["error"]["message"]
    assert "backend boom" in payload["error"]["detail"]


@pytest.mark.parametrize("source_type", ["metadata_csv", "local_dir"])
def test_ingest_accepts_empty_path_for_default_sources(
    monkeypatch: pytest.MonkeyPatch,
    api_root: Path,
    source_type: str,
) -> None:
    captured: dict[str, object] = {}

    class FakeSource:
        def discover(self) -> list[DocumentMeta]:
            return []

    def fake_create_source(raw_source_type: str, path: Path | None) -> FakeSource:
        captured["source_type"] = raw_source_type
        captured["path"] = path
        return FakeSource()

    monkeypatch.setattr(sources_module, "create_source", fake_create_source)
    monkeypatch.setattr(routes_module, "_run_ingest_job", lambda source, path: None)
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/ingest", json={"source": source_type})

    assert response.status_code == 202
    assert captured == {"source_type": source_type, "path": None}
    assert response.json()["status"] == "accepted"


def test_ingest_url_list_requires_path(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/ingest", json={"source": "url_list"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["message"]
    assert "path is required when source=url_list" in payload["error"]["detail"]


def test_ingest_endpoint_accepts_request_and_schedules_background_job(
    monkeypatch: pytest.MonkeyPatch,
    api_root: Path,
) -> None:
    captured: dict[str, object] = {}
    scheduled_jobs: list[tuple[str, str]] = []

    class FakeSource:
        def discover(self) -> list[DocumentMeta]:
            return [
                DocumentMeta(doc_id="paper1", source_type="local_dir", filename="paper1.pdf"),
                DocumentMeta(doc_id="paper2", source_type="local_dir", filename="paper2.pdf"),
            ]

    def fake_create_source(source_type: str, path: Path | None) -> FakeSource:
        captured["source_type"] = source_type
        captured["path"] = path
        return FakeSource()

    monkeypatch.setattr(sources_module, "create_source", fake_create_source)
    monkeypatch.setattr(routes_module, "_run_ingest_job", lambda source, path: scheduled_jobs.append((source, path)))
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/ingest", json={"source": "local_dir", "path": "/tmp/new_papers"})

    assert response.status_code == 202
    assert captured == {
        "source_type": "local_dir",
        "path": Path("/tmp/new_papers"),
    }
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["message"] == "Ingestion started for 2 documents"
    assert payload["task_id"].startswith("ingest_")
    assert scheduled_jobs == [("local_dir", "/tmp/new_papers")]


def test_ingest_missing_source_returns_error_schema(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    def fake_create_source(source_type: str, path: Path | None) -> None:
        raise FileNotFoundError("missing source path")

    monkeypatch.setattr(sources_module, "create_source", fake_create_source)
    client = _make_client(monkeypatch, api_root)

    response = client.post("/api/v1/ingest", json={"source": "local_dir", "path": "/tmp/missing"})

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "SOURCE_NOT_FOUND"
    assert payload["error"]["message"]
    assert "missing source path" in payload["error"]["detail"]


def test_unknown_route_uses_uniform_error_schema(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    client = _make_client(monkeypatch, api_root)

    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "detail"}
    assert payload["error"]["detail"]


def test_method_not_allowed_uses_uniform_error_schema(monkeypatch: pytest.MonkeyPatch, api_root: Path) -> None:
    client = _make_client(monkeypatch, api_root)

    response = client.get("/api/v1/search")

    assert response.status_code == 405
    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "detail"}
    assert payload["error"]["detail"]


@pytest.mark.parametrize(
    ("path", "method", "success_status", "error_statuses"),
    [
        ("/api/v1/search", "post", "200", {"422", "503", "500"}),
        ("/api/v1/query", "post", "200", {"422", "503", "500"}),
        ("/api/v1/ingest", "post", "202", {"404", "422", "500"}),
    ],
)
def test_openapi_documents_actual_error_responses(
    monkeypatch: pytest.MonkeyPatch,
    api_root: Path,
    path: str,
    method: str,
    success_status: str,
    error_statuses: set[str],
) -> None:
    client = _make_client(monkeypatch, api_root)

    openapi = client.get("/openapi.json").json()
    responses = openapi["paths"][path][method]["responses"]

    assert success_status in responses
    for status_code in error_statuses:
        assert status_code in responses
        assert responses[status_code]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }
