from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer
import uvicorn

from src.api.app import create_app
from src.config.settings import get_settings
from src.core.logging import get_logger, report_error
from src.core.model_assets import ensure_parse_runtime_dependencies
from src.core.schemas import SearchResult
from src.evaluation.evaluator import Evaluator
from src.generation.llm_client import ensure_ollama_ready
from src.generation.rag_pipeline import RAGConfig, RAGPipeline
from src.ingestion.pipeline import run_ingest
from src.retrieval import get_retriever
from src.retrieval.reranker import Reranker

app = typer.Typer(help="RAG command line interface")
logger = get_logger(__name__)


def _retrieve_results(question: str, top_k: int, retrieval_mode: str, rerank: bool = False) -> list[dict[str, Any]]:
    retriever = get_retriever(retrieval_mode)
    fetch_k = top_k * 3 if rerank else top_k
    results = retriever.retrieve(question, fetch_k)
    if not rerank:
        return results[:top_k]
    reranker = Reranker()
    return reranker.rerank(question, results, top_k)


def _missing_env_message(command_name: str, missing_vars: list[str]) -> str:
    missing = ", ".join(missing_vars)
    return (
        f"{command_name} requires {missing}. "
        "Please copy .env.example to .env and fill in the missing values."
    )


def _ensure_env_configured(command_name: str, checks: list[tuple[str, str]]) -> None:
    missing_vars = [env_name for env_name, value in checks if not value.strip()]
    if missing_vars:
        raise typer.BadParameter(_missing_env_message(command_name, missing_vars))


def _resolve_api_generation_options(model: str | None) -> tuple[str, str]:
    settings = get_settings()
    _ensure_env_configured(
        "rag query --llm api",
        [
            ("RAG_API_BASE_URL", settings.api_base_url),
            ("RAG_API_KEY", settings.api_key),
        ],
    )
    api_key = settings.api_key.strip()
    effective_model = model.strip() if model is not None and model.strip() else settings.api_model

    return api_key, effective_model


def _ensure_parse_api_configured() -> None:
    settings = get_settings()
    _ensure_env_configured(
        "rag parse --llm api",
        [
            ("RAG_VISUAL_API_BASE_URL or RAG_API_BASE_URL", settings.visual_api_base_url or settings.api_base_url),
            ("RAG_VISUAL_API_KEY or RAG_API_KEY", settings.visual_api_key or settings.api_key),
        ],
    )


@app.command(name="parse")
def parse(
    source: str = typer.Option("local_dir", "--source", help="Source type: local_dir | url_csv | url_list"),
    path: Path | None = typer.Option(None, "--path", help="Path to source"),
    device: str = typer.Option("cpu", "--device", help="MinerU device: cpu | cuda | cuda:0 | npu | mps"),
    llm: str = typer.Option("api", "--llm", help="LLM backend for visual summary: api | ollama"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index", help="Force rebuild FAISS + BM25"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Retry previously failed docs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the planned ingest actions only"),
) -> None:
    if llm not in {"api", "ollama"}:
        report_error(logger, "parse_invalid_llm_backend", "Invalid LLM backend", llm_backend=llm)
        raise typer.Exit(code=2)

    if not dry_run:
        try:
            typer.echo("Preparing parse runtime models...", err=True)
            ensure_parse_runtime_dependencies()
            typer.echo("Runtime models are ready.", err=True)
        except Exception as exc:
            typer.echo(str(exc), err=True)
            report_error(logger, "parse_runtime_prepare_failed", "Parse runtime setup failed", error=str(exc))
            raise typer.Exit(code=2) from exc

    if llm == "api":
        try:
            typer.echo("Validating API configuration...", err=True)
            _ensure_parse_api_configured()
        except typer.BadParameter as exc:
            typer.echo(str(exc), err=True)
            report_error(logger, "parse_prompt_failed", "Parse setup failed", error=str(exc))
            raise typer.Exit(code=2) from exc

    try:
        if llm == "ollama":
            typer.echo("Checking Ollama service...", err=True)
            ensure_ollama_ready()
            typer.echo("Ollama is ready.", err=True)
        typer.echo("Starting parse pipeline...", err=True)
        summary = run_ingest(
            source=source,
            path=path,
            device=device,
            llm_backend=llm,
            rebuild_index=rebuild_index,
            retry_failed=retry_failed,
            dry_run=dry_run,
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        report_error(logger, "ingest_failed", "Parse failed", error=str(exc))
        raise typer.Exit(code=2) from exc

    typer.echo(json.dumps(summary, ensure_ascii=False))
    if summary.get("status") not in (None, "ok"):
        raise typer.Exit(code=1)


@app.command()
def query(
    question: str = typer.Argument(..., metavar="QUESTION", help="The question string"),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve"),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="dense | sparse | hybrid"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable cross-encoder reranking"),
    llm: str = typer.Option("api", "--llm", help="LLM backend: api | ollama"),
    model: str | None = typer.Option(None, "--model", help="Generation model when llm=api"),
    output: Path | None = typer.Option(None, "--output", help="Write JSON answer to file instead of stdout"),
) -> None:
    settings = get_settings()
    api_key: str | None = None
    llm_model: str | None = None

    if llm not in {"api", "ollama"}:
        report_error(logger, "query_invalid_llm_backend", "Invalid LLM backend", llm_backend=llm)
        raise typer.Exit(code=2)

    if llm == "api":
        try:
            api_key, llm_model = _resolve_api_generation_options(model)
        except typer.BadParameter as exc:
            typer.echo(str(exc), err=True)
            report_error(logger, "query_prompt_failed", "Query setup failed", error=str(exc))
            raise typer.Exit(code=2) from exc
    else:
        if model is not None:
            report_error(logger, "query_invalid_model_option", "Model option is invalid for ollama", llm_backend=llm)
            raise typer.Exit(code=2)

    try:
        if llm == "ollama":
            ensure_ollama_ready()
            typer.echo("Ready for questions", err=True)

        pipeline = RAGPipeline(
            config=RAGConfig(
                top_k=top_k,
                retrieval_mode=retrieval_mode or settings.retrieval_mode,
                rerank=rerank,
                llm_backend=llm,
                llm_model=(llm_model or settings.api_model) if llm == "api" else None,
                api_key=api_key,
            )
        )
        answer = pipeline.answer_question(
            question,
            top_k=top_k,
            retrieval_mode=retrieval_mode or settings.retrieval_mode,
            rerank=rerank,
            llm_backend=llm,
            llm_model=(llm_model or settings.api_model) if llm == "api" else None,
            api_key=api_key,
        )
    except Exception as exc:
        report_error(logger, "query_generation_failed", "Query failed", error=str(exc))
        raise typer.Exit(code=1) from exc

    payload = json.dumps(answer.model_dump(mode="json"), ensure_ascii=False)
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

    try:
        start = time.perf_counter()
        hits = _retrieve_results(question, top_k, effective_mode, rerank=rerank)
        retrieval_latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        report_error(logger, "search_failed", "Search failed", error=str(exc))
        raise typer.Exit(code=1) from exc

    results = [
        SearchResult(
            rank=int(hit.get("rank", index)),
            doc_id=str(hit.get("doc_id", "")),
            chunk_id=str(hit.get("chunk_id", "")),
            chunk_type=str(hit.get("chunk_type", "text")),
            score=float(hit.get("score", 0.0)),
            text=str(hit.get("text", "")),
            page_number=hit.get("page_number"),
            headings=list(hit.get("headings", []) or []),
            caption=str(hit.get("caption", "")),
            asset_path=str(hit.get("asset_path", "")),
        )
        for index, hit in enumerate(hits, start=1)
    ]

    if output_format == "table":
        typer.echo("rank\tdoc_id\tchunk_id\ttype\tscore\tpage\tcaption\ttext")
        for item in results:
            typer.echo(
                f"{item.rank}\t{item.doc_id}\t{item.chunk_id}\t{item.chunk_type.value}\t"
                f"{item.score:.6f}\t{item.page_number}\t{item.caption[:60]}\t{item.text[:120]}"
            )
        return

    typer.echo(
        json.dumps(
            {
                "results": [result.model_dump(mode="json") for result in results],
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                "retrieval_mode": effective_mode,
                "total_results": len(results),
            },
            ensure_ascii=False,
        )
    )


@app.command()
def benchmark(
    dataset: Path = typer.Option(Path("data/benchmark_QA.csv"), "--dataset", help="Path to labeled Q&A CSV"),
    retrieval_mode: str = typer.Option("hybrid", "--retrieval-mode", help="dense | sparse | hybrid"),
    top_k: int = typer.Option(5, "--top-k"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    output_dir: Path = typer.Option(Path("data/benchmark_results/"), "--output-dir", help="Directory for report output"),
    tag: str = typer.Option("run", "--tag", help="Human-readable tag for this run"),
) -> None:
    evaluator = Evaluator(
        dataset=dataset,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        rerank=rerank,
        output_dir=output_dir,
        tag=tag,
    )
    report_path, summary_path = evaluator.run()
    typer.echo(json.dumps({"report": str(report_path), "summary": str(summary_path)}, ensure_ascii=False))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    if reload:
        uvicorn.run("src.api.app:create_app", factory=True, host=host, port=port, reload=True)
        return

    uvicorn.run(create_app(), host=host, port=port, reload=False)
