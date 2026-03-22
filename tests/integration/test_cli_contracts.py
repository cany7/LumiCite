from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import src.main as main_module
from src.core.schemas import Citation, ChunkType, RAGAnswer, VerificationResult


def _answer() -> RAGAnswer:
    return RAGAnswer(
        answer="552 tCO2e",
        supporting_materials="alpha evidence",
        explanation="grounded answer",
        citations=[
            Citation(
                doc_id="paper1",
                chunk_id="paper1_deadbeef",
                page_number=1,
                evidence_text="alpha evidence",
                evidence_type=ChunkType.TEXT,
            )
        ],
        retrieval_latency_ms=12.0,
        generation_latency_ms=34.0,
        retrieval_mode="hybrid",
        llm_backend="api",
        verification=VerificationResult(passed=True, confidence=0.9, warnings=[]),
    )


def test_parse_command_uses_documented_defaults(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_ingest(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"status": "ok", "event": "parse_summary"}

    monkeypatch.setattr(main_module, "run_ingest", fake_run_ingest)
    monkeypatch.setattr(main_module, "ensure_parse_runtime_dependencies", lambda: None)
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            visual_api_key="visual-key",
            api_key="",
            visual_api_base_url="https://api.example.com/v1",
            api_base_url="",
        ),
    )

    result = runner.invoke(main_module.app, ["parse"])

    assert result.exit_code == 0
    assert captured == {
        "source": "local_dir",
        "path": None,
        "device": "cpu",
        "llm_backend": "api",
        "rebuild_index": False,
        "retry_failed": False,
        "dry_run": False,
    }
    assert json.loads(result.stdout)["status"] == "ok"


def test_parse_command_waits_for_ollama_when_requested(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {"ready": False}

    def fake_run_ingest(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"status": "ok", "event": "parse_summary"}

    monkeypatch.setattr(main_module, "ensure_parse_runtime_dependencies", lambda: None)
    monkeypatch.setattr(main_module, "ensure_ollama_ready", lambda: captured.update({"ready": True}))
    monkeypatch.setattr(main_module, "run_ingest", fake_run_ingest)

    result = runner.invoke(main_module.app, ["parse", "--llm", "ollama"])

    assert result.exit_code == 0
    assert captured["ready"] is True
    assert captured["llm_backend"] == "ollama"


def test_search_command_supports_json_and_table_output(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(main_module, "get_settings", lambda: SimpleNamespace(retrieval_mode="hybrid"))
    monkeypatch.setattr(
        main_module,
        "_retrieve_results",
        lambda question, top_k, retrieval_mode, rerank: [
            {
                "rank": 1,
                "doc_id": "paper2",
                "chunk_id": "paper2_fig_cafebabe",
                "chunk_type": "figure",
                "score": 0.9,
                "text": "figure summary",
                "page_number": 2,
                "headings": ["Results"],
                "caption": "Figure 2. Trend",
                "asset_path": "data/assets/paper2/paper2_fig_cafebabe.png",
            }
        ],
    )

    json_result = runner.invoke(main_module.app, ["search", "trend"])
    table_result = runner.invoke(main_module.app, ["search", "trend", "--output-format", "table"])

    assert json_result.exit_code == 0
    assert json.loads(json_result.stdout)["results"][0]["chunk_type"] == "figure"
    assert table_result.exit_code == 0
    assert "rank\tdoc_id\tchunk_id\ttype\tscore\tpage\tcaption\ttext" in table_result.stdout
    assert "paper2_fig_cafebabe" in table_result.stdout


def test_query_command_defaults_to_api_and_allows_model_override(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, config) -> None:  # noqa: ANN001
            captured["config"] = config

        def answer_question(self, question: str, **kwargs):  # noqa: ANN003
            captured["question"] = question
            captured["kwargs"] = kwargs
            return _answer()

    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            api_key="test-api-key",
            api_base_url="https://api.example.com/v1",
            api_model="qwen/default",
            retrieval_mode="hybrid",
        ),
    )
    monkeypatch.setattr(main_module, "RAGPipeline", FakePipeline)

    result = runner.invoke(main_module.app, ["query", "What happened?", "--model", "qwen/custom"])

    assert result.exit_code == 0
    assert captured["config"].llm_backend == "api"
    assert captured["config"].llm_model == "qwen/custom"
    assert captured["config"].api_key == "test-api-key"
    assert captured["kwargs"]["llm_backend"] == "api"
    assert captured["kwargs"]["llm_model"] == "qwen/custom"
    assert json.loads(result.stdout)["llm_backend"] == "api"
    assert "answer_value" not in json.loads(result.stdout)
    assert "answer_unit" not in json.loads(result.stdout)


def test_query_command_requires_env_configuration_for_api_backend(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(api_key="", api_base_url="", api_model="qwen/default", retrieval_mode="hybrid"),
    )

    result = runner.invoke(main_module.app, ["query", "What happened?"])

    assert result.exit_code == 2
    assert "Please copy .env.example to .env" in result.output
    assert "RAG_API_BASE_URL" in result.output
    assert "RAG_API_KEY" in result.output


def test_parse_command_requires_env_configuration_for_api_backend(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(main_module, "ensure_parse_runtime_dependencies", lambda: None)
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            visual_api_key="",
            api_key="",
            visual_api_base_url="",
            api_base_url="",
        ),
    )

    result = runner.invoke(main_module.app, ["parse"])

    assert result.exit_code == 2
    assert "Please copy .env.example to .env" in result.output
    assert "RAG_VISUAL_API_BASE_URL or RAG_API_BASE_URL" in result.output
    assert "RAG_VISUAL_API_KEY or RAG_API_KEY" in result.output


def test_parse_command_prepares_runtime_dependencies_before_running(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {"prepared": False}

    def fake_prepare() -> None:
        captured["prepared"] = True

    def fake_run_ingest(**kwargs):  # noqa: ANN003
        captured["llm_backend"] = kwargs["llm_backend"]
        return {"status": "ok", "event": "parse_summary"}

    monkeypatch.setattr(main_module, "ensure_parse_runtime_dependencies", fake_prepare)
    monkeypatch.setattr(main_module, "run_ingest", fake_run_ingest)
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: SimpleNamespace(
            visual_api_key="visual-key",
            api_key="",
            visual_api_base_url="https://api.example.com/v1",
            api_base_url="",
        ),
    )

    result = runner.invoke(main_module.app, ["parse"])

    assert result.exit_code == 0
    assert captured["prepared"] is True
    assert captured["llm_backend"] == "api"


def test_query_command_waits_for_ollama_when_requested(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {"ready": False}

    class FakePipeline:
        def __init__(self, config) -> None:  # noqa: ANN001
            captured["config"] = config

        def answer_question(self, question: str, **kwargs):  # noqa: ANN003
            return _answer().model_copy(update={"llm_backend": "ollama"})

    monkeypatch.setattr(main_module, "get_settings", lambda: SimpleNamespace(retrieval_mode="hybrid", api_model="qwen/default"))
    monkeypatch.setattr(main_module, "ensure_ollama_ready", lambda: captured.update({"ready": True}))
    monkeypatch.setattr(main_module, "RAGPipeline", FakePipeline)

    result = runner.invoke(main_module.app, ["query", "What happened?", "--llm", "ollama"])

    assert result.exit_code == 0
    assert captured["ready"] is True
    assert captured["config"].llm_backend == "ollama"
    assert captured["config"].llm_model is None
    assert "Ready for questions" in result.output
    assert json.loads(result.stdout.splitlines()[-1])["llm_backend"] == "ollama"


def test_benchmark_and_serve_commands_delegate_to_current_entrypoints(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.md"
    captured: dict[str, object] = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):  # noqa: ANN003
            captured["benchmark_kwargs"] = kwargs

        def run(self) -> tuple[Path, Path]:
            return report_path, summary_path

    def fake_uvicorn_run(target, **kwargs):  # noqa: ANN003
        captured.setdefault("uvicorn_calls", []).append((target, kwargs))

    monkeypatch.setattr(main_module, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(main_module.uvicorn, "run", fake_uvicorn_run)

    benchmark_result = runner.invoke(main_module.app, ["benchmark", "--tag", "phase4"])
    serve_reload_result = runner.invoke(main_module.app, ["serve", "--reload"])
    serve_plain_result = runner.invoke(main_module.app, ["serve"])

    assert benchmark_result.exit_code == 0
    assert json.loads(benchmark_result.stdout) == {"report": str(report_path), "summary": str(summary_path)}
    assert captured["benchmark_kwargs"]["tag"] == "phase4"

    assert serve_reload_result.exit_code == 0
    assert serve_plain_result.exit_code == 0
    assert captured["uvicorn_calls"][0][0] == "src.api.app:create_app"
    assert captured["uvicorn_calls"][0][1]["factory"] is True
    assert callable(captured["uvicorn_calls"][1][0])
