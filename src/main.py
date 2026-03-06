from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import typer
import uvicorn
from fastapi import FastAPI

from src.config.settings import get_settings
from src.core.constants import FALLBACK_ANSWER
from src.core.logging import get_logger

app = typer.Typer(help="RAG command line interface")
logger = get_logger(__name__)


def _now_utc() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _fallback_answer(retrieval_mode: str, llm_backend: str) -> dict:
    return {
        "answer": FALLBACK_ANSWER,
        "answer_value": "is_blank",
        "answer_unit": "is_blank",
        "ref_id": [],
        "supporting_materials": "is_blank",
        "explanation": "is_blank",
        "citations": [],
        "retrieval_latency_ms": 0.1,
        "generation_latency_ms": 0.1,
        "retrieval_mode": retrieval_mode,
        "llm_backend": llm_backend,
        "verification": {
            "passed": False,
            "confidence": 0.0,
            "warnings": ["Query path is a Phase 1 fallback implementation."],
            "corrected_output": None,
        },
    }


@app.command()
def ingest(
    source: str = typer.Option("metadata_csv", "--source", help="Source type: metadata_csv | local_dir | url_list"),
    path: Path | None = typer.Option(None, "--path", help="Path to source (CSV file, directory, or URL list file)"),
    workers: int | None = typer.Option(None, "--workers", help="Parallel workers for PDF processing"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index", help="Force rebuild FAISS + BM25 index after ingest"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Re-process only previously failed documents"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed, do nothing"),
) -> None:
    settings = get_settings()
    effective_workers = workers if workers is not None else settings.ingest_workers
    payload = {
        "status": "accepted" if not dry_run else "dry_run",
        "source": source,
        "path": str(path) if path else "",
        "workers": effective_workers,
        "rebuild_index": rebuild_index,
        "retry_failed": retry_failed,
        "dry_run": dry_run,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False))


@app.command()
def query(
    question: str = typer.Argument(..., metavar="QUESTION", help="The question string"),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve"),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="dense | sparse | hybrid"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable cross-encoder reranking"),
    llm: str | None = typer.Option(None, "--llm", help="LLM backend: gemini | ollama"),
    output: Path | None = typer.Option(None, "--output", help="Write JSON answer to file instead of stdout"),
) -> None:
    settings = get_settings()
    effective_mode = retrieval_mode or settings.retrieval_mode
    effective_llm = llm or settings.llm_backend

    result = _fallback_answer(retrieval_mode=effective_mode, llm_backend=effective_llm)

    logger.info("timed", operation="retrieve", latency_ms=result["retrieval_latency_ms"], top_k=top_k, rerank=rerank)
    logger.info("timed", operation="generate", latency_ms=result["generation_latency_ms"], llm_backend=effective_llm)

    payload = json.dumps(result, ensure_ascii=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    else:
        typer.echo(payload)


@app.command()
def search(
    question: str = typer.Argument(..., metavar="QUESTION", help="The query string"),
    top_k: int = typer.Option(10, "--top-k", help="Number of results"),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="dense | sparse | hybrid"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Enable reranking"),
    output_format: str = typer.Option("json", "--output-format", help="json | table"),
) -> None:
    settings = get_settings()
    effective_mode = retrieval_mode or settings.retrieval_mode
    data = {
        "results": [],
        "retrieval_latency_ms": 0.1,
        "retrieval_mode": effective_mode,
        "total_results": 0,
    }
    if output_format == "table":
        typer.echo("rank\tref_id\tscore\ttext")
    else:
        typer.echo(json.dumps(data, ensure_ascii=False))
    logger.info("timed", operation="retrieve", latency_ms=data["retrieval_latency_ms"], query=question, top_k=top_k, rerank=rerank)


@app.command()
def benchmark(
    dataset: Path = typer.Option(Path("data/train_QA.csv"), "--dataset", help="Path to labeled Q&A CSV"),
    retrieval_mode: str = typer.Option("hybrid", "--retrieval-mode", help="dense | sparse | hybrid"),
    top_k: int = typer.Option(5, "--top-k"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    output_dir: Path = typer.Option(Path("data/benchmark_results/"), "--output-dir", help="Directory for report output"),
    tag: str = typer.Option("run", "--tag", help="Human-readable tag for this run"),
) -> None:
    settings = get_settings()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"{tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": run_id,
        "tag": tag,
        "timestamp": _now_utc(),
        "config_hash": hashlib.md5(f"{retrieval_mode}:{top_k}:{rerank}".encode("utf-8")).hexdigest(),
        "git_commit": "",
        "dataset": str(dataset),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "reranker_enabled": rerank,
        "embedding_model": settings.embedding_model,
        "num_questions": 0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
        "mean_retrieval_latency_ms": 0.0,
        "p95_retrieval_latency_ms": 0.0,
        "per_question": [],
    }

    report_path = output_dir / f"{run_id}_report.json"
    summary_path = output_dir / f"{run_id}_summary.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "retrieval_mode", "top_k", "rerank", "recall_at_k", "mrr", "ndcg_at_k"])
        writer.writerow([run_id, retrieval_mode, top_k, int(rerank), 0.0, 0.0, 0.0])

    typer.echo(json.dumps({"report": str(report_path), "summary": str(summary_path)}, ensure_ascii=False))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    api = FastAPI(title="RAG API")

    @api.get("/api/v1/health")
    def health() -> dict:
        settings = get_settings()
        return {
            "status": "ok",
            "index_loaded": False,
            "num_papers": 0,
            "num_chunks": 0,
            "embedding_model": settings.embedding_model,
            "retrieval_modes_available": ["dense", "sparse", "hybrid"],
        }

    uvicorn.run(api, host=host, port=port, reload=reload)
