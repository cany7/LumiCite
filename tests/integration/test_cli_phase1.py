from __future__ import annotations

import json

from typer.testing import CliRunner

from src.config.settings import get_settings
from src.core.schemas import TextChunk
from src.main import app
import src.main as main_module

runner = CliRunner()


def test_query_stdout_conforms_to_phase1_contract(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "_retrieve_results",
        lambda question, top_k, retrieval_mode, rerank=False: [
            {
                "rank": 1,
                "chunk_id": "paper1_deadbeef",
                "ref_id": "paper1",
                "score": 0.9,
                "text": "evidence text",
                "page": 2,
                "source_file": "paper1.pdf",
                "headings": ["Intro"],
                "chunk_type": "text",
            }
        ],
    )

    class FakeGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate_json(self, prompt):
            return {
                "answer": "42",
                "answer_value": "42",
                "answer_unit": "units",
                "ref_id": ["paper1"],
                "supporting_materials": "evidence text",
                "explanation": "derived from evidence",
            }

    monkeypatch.setattr(main_module, "LLMGenerator", FakeGenerator)

    result = runner.invoke(app, ["query", "What is the answer?"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"] == "42"
    assert payload["ref_id"] == ["paper1"]
    assert payload["retrieval_mode"] in {"dense", "sparse", "hybrid"}
    assert payload["llm_backend"] in {"gemini", "ollama"}
    assert isinstance(payload.get("retrieval_latency_ms"), (int, float, type(None)))
    assert isinstance(payload.get("generation_latency_ms"), (int, float, type(None)))
    assert payload["citations"][0]["ref_id"] == "paper1"


def test_query_output_writes_json_file(monkeypatch, tmp_path):
    out_path = tmp_path / "answer.json"
    monkeypatch.setattr(main_module, "_retrieve_results", lambda question, top_k, retrieval_mode, rerank=False: [])

    result = runner.invoke(app, ["query", "What is the answer?", "--output", str(out_path)])

    assert result.exit_code == 0
    assert result.stdout == ""
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["ref_id"], list)


def test_search_json_contract_uses_settings_default(monkeypatch):
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "sparse")
    get_settings.cache_clear()
    monkeypatch.setattr(
        main_module,
        "_retrieve_results",
        lambda question, top_k, retrieval_mode, rerank=False: [
            {
                "rank": 1,
                "chunk_id": "paper1_deadbeef",
                "ref_id": "paper1",
                "score": 0.7,
                "text": "match text",
                "page": 5,
                "source_file": "paper1.pdf",
                "headings": ["Results"],
                "chunk_type": "text",
            }
        ],
    )

    result = runner.invoke(app, ["search", "test question"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["retrieval_mode"] == "sparse"
    assert payload["total_results"] == 1
    assert payload["results"][0]["ref_id"] == "paper1"
    assert payload["results"][0]["page"] == 5
    assert isinstance(payload.get("retrieval_latency_ms"), (int, float))


def test_search_table_output_returns_text_response(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "_retrieve_results",
        lambda question, top_k, retrieval_mode, rerank=False: [
            {
                "rank": 1,
                "chunk_id": "paper1_deadbeef",
                "ref_id": "paper1",
                "score": 0.7,
                "text": "match text",
                "page": 5,
                "source_file": "paper1.pdf",
                "headings": ["Results"],
                "chunk_type": "text",
            }
        ],
    )

    result = runner.invoke(app, ["search", "test question", "--output-format", "table"])

    assert result.exit_code == 0
    assert "paper1" in result.stdout
    assert "match text" in result.stdout


def test_ingest_uses_settings_default_workers_and_writes_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_INGEST_WORKERS", "9")
    get_settings.cache_clear()
    monkeypatch.setattr(main_module, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(main_module, "_discover_pdfs", lambda source, path: [tmp_path / "paper1.pdf"])
    monkeypatch.setattr(
        main_module,
        "extract_pdf_chunks",
        lambda pdf_path: [TextChunk(chunk_id="paper1_deadbeef", doc_id="paper1", text="alpha")],
    )
    monkeypatch.setattr(main_module, "embed_chunks", lambda records, model_name, batch_size: [[0.1, 0.2]])
    monkeypatch.setattr(
        main_module,
        "_build_search_indices",
        lambda chunks_path, embeddings_path, embedding_model: {
            "faiss_index": str(tmp_path / "data" / "my_faiss.index"),
            "faiss_metadata": str(tmp_path / "data" / "my_faiss.meta.json"),
            "text_data": str(tmp_path / "data" / "text_data.pkl"),
            "vectors": 1,
            "bm25_index": str(tmp_path / "data" / "bm25_index.pkl"),
            "bm25_metadata": str(tmp_path / "data" / "bm25_index.meta.json"),
            "bm25_rows": 1,
        },
    )

    result = runner.invoke(app, ["ingest", "--source", "local_dir"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["workers"] == 9
    assert payload["chunks_written"] == 1
    assert payload["embeddings_written"] == 1
    assert payload["vectors"] == 1
    assert payload["bm25_rows"] == 1
    assert payload["indices_rebuilt"] is True
    assert (tmp_path / "data" / "JSON" / "chunks.jsonl").exists()
    assert (tmp_path / "data" / "JSON" / "embeddings.jsonl").exists()


def test_benchmark_writes_report_and_summary_files(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("question\nwhat is test?\n", encoding="utf-8")

    class FakeEvaluator:
        def __init__(self, dataset, retrieval_mode, top_k, rerank, output_dir, tag):
            self.output_dir = output_dir
            self.tag = tag

        def run(self):
            report_path = self.output_dir / "phase1_20260305_120000_report.json"
            summary_path = self.output_dir / "phase1_20260305_120000_summary.csv"
            report_path.write_text("{}", encoding="utf-8")
            summary_path.write_text("run_id\nphase1\n", encoding="utf-8")
            return report_path, summary_path

    monkeypatch.setattr(main_module, "Evaluator", FakeEvaluator)

    result = runner.invoke(app, ["benchmark", "--dataset", str(dataset), "--output-dir", str(tmp_path), "--tag", "phase1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    report_path = tmp_path / next(path.name for path in tmp_path.glob("phase1_*_report.json"))
    summary_path = tmp_path / next(path.name for path in tmp_path.glob("phase1_*_summary.csv"))
    assert payload["report"] == str(report_path)
    assert payload["summary"] == str(summary_path)
