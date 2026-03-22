from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer
import uvicorn

from src.api.app import create_app
from src.config.settings import get_settings, normalize_reasoning_effort
from src.core.constants import FALLBACK_ANSWER
from src.core.logging import get_logger, report_error
from src.core.model_assets import ensure_parse_runtime_dependencies
from src.core.schemas import RAGAnswer, SearchResult
from src.evaluation.evaluator import Evaluator
from src.generation.llm_client import ensure_ollama_ready
from src.generation.rag_pipeline import RAGConfig, RAGPipeline
from src.ingestion.pipeline import run_ingest
from src.retrieval.query_explanation import QueryExplanationConfig, retrieve_with_optional_query_explanation

app = typer.Typer(help="RAG command line interface")
logger = get_logger(__name__)
REASONING_EFFORT_OPTIONS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _retrieve_results(
    question: str,
    top_k: int,
    retrieval_mode: str,
    rerank: bool = False,
    *,
    query_explanation: QueryExplanationConfig | None = None,
) -> list[dict[str, Any]]:
    return retrieve_with_optional_query_explanation(
        question,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        rerank=rerank,
        query_explanation=query_explanation,
    ).results


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


def _single_line(text: str) -> str:
    return " ".join(str(text).split())


def _truncate_text(text: str, limit: int = 160) -> str:
    normalized = _single_line(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _render_query_text(question: str, answer: RAGAnswer) -> str:
    blocks = [
        ("Question", question.strip()),
        ("Answer", answer.answer.strip()),
    ]

    if answer.answer != FALLBACK_ANSWER:
        blocks.append(("Evidence", answer.supporting_materials.strip()))
        citation_lines = [
            f"- {citation.evidence_type.value} | {citation.chunk_id} | {_truncate_text(citation.evidence_text)}"
            for citation in answer.citations
        ]
        blocks.append(("Citations", "\n".join(citation_lines) if citation_lines else "- None"))

    rendered_blocks: list[str] = []
    for title, body in blocks:
        rendered_blocks.append(f"{title}\n{body}")
    return "\n\n".join(rendered_blocks)


def _ensure_parse_api_configured() -> None:
    settings = get_settings()
    _ensure_env_configured(
        "rag parse --llm api",
        [
            ("RAG_VISUAL_API_BASE_URL or RAG_API_BASE_URL", settings.visual_api_base_url or settings.api_base_url),
            ("RAG_VISUAL_API_KEY or RAG_API_KEY", settings.visual_api_key or settings.api_key),
        ],
    )


def _ensure_query_explanation_api_configured() -> None:
    settings = get_settings()
    _ensure_env_configured(
        "rag query --query-explanation",
        [
            ("RAG_API_BASE_URL", settings.api_base_url),
            ("RAG_API_KEY", settings.api_key),
        ],
    )


def _resolve_retrieval_top_k(top_k: int | None) -> int:
    settings = get_settings()
    return settings.retrieval_top_k if top_k is None else top_k


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
    top_k: int | None = typer.Option(None, "--top-k", help="Number of chunks to retrieve"),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="dense | sparse | hybrid"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Enable cross-encoder reranking"),
    query_explanation: bool = typer.Option(
        True,
        "--query-explanation/--no-query-explanation",
        help="Enable retrieval-oriented query expansion before retrieval",
    ),
    llm: str = typer.Option("api", "--llm", help="LLM backend: api | ollama"),
    model: str | None = typer.Option(None, "--model", help="Generation model when llm=api"),
    reasoning_effort: str = typer.Option(
        "none",
        "--reasoning-effort",
        help="Reasoning effort for API backends: none | minimal | low | medium | high | xhigh",
    ),
    output: Path | None = typer.Option(None, "--output", help="Write the answer to file instead of stdout"),
    output_format: str = typer.Option("text", "--output-format", help="text | json"),
    json_output: bool = typer.Option(False, "--json", help="Alias for --output-format json"),
) -> None:
    settings = get_settings()
    effective_top_k = _resolve_retrieval_top_k(top_k)
    api_key: str | None = None
    llm_model: str | None = None
    effective_output_format = "json" if json_output else output_format
    raw_reasoning_effort = reasoning_effort.strip().lower()
    effective_reasoning_effort = normalize_reasoning_effort(raw_reasoning_effort)

    if llm not in {"api", "ollama"}:
        report_error(logger, "query_invalid_llm_backend", "Invalid LLM backend", llm_backend=llm)
        raise typer.Exit(code=2)
    if effective_output_format not in {"text", "json"}:
        report_error(logger, "query_invalid_output_format", "Invalid output format", output_format=effective_output_format)
        raise typer.Exit(code=2)
    if raw_reasoning_effort not in REASONING_EFFORT_OPTIONS:
        report_error(
            logger,
            "query_invalid_reasoning_effort",
            "Invalid reasoning effort",
            reasoning_effort=raw_reasoning_effort,
        )
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
        if raw_reasoning_effort != "none":
            report_error(
                logger,
                "query_invalid_reasoning_effort_for_ollama",
                "Reasoning effort is invalid for ollama",
                llm_backend=llm,
                reasoning_effort=raw_reasoning_effort,
            )
            raise typer.Exit(code=2)
        if query_explanation:
            try:
                _ensure_query_explanation_api_configured()
            except typer.BadParameter as exc:
                typer.echo(str(exc), err=True)
                report_error(logger, "query_explanation_setup_failed", "Query setup failed", error=str(exc))
                raise typer.Exit(code=2) from exc

    try:
        if llm == "ollama":
            typer.echo("Checking Ollama service...", err=True)
            ensure_ollama_ready()
            typer.echo("Ollama is ready.", err=True)

        typer.echo("Running query pipeline...", err=True)
        pipeline = RAGPipeline(
            config=RAGConfig(
                top_k=effective_top_k,
                retrieval_mode=retrieval_mode or settings.retrieval_mode,
                rerank=rerank,
                query_explanation_enabled=query_explanation,
                llm_backend=llm,
                llm_model=(llm_model or settings.api_model) if llm == "api" else None,
                api_key=api_key,
                reasoning_effort=effective_reasoning_effort if llm == "api" else None,
            )
        )
        answer = pipeline.answer_question(
            question,
            top_k=effective_top_k,
            retrieval_mode=retrieval_mode or settings.retrieval_mode,
            rerank=rerank,
            query_explanation_enabled=query_explanation,
            llm_backend=llm,
            llm_model=(llm_model or settings.api_model) if llm == "api" else None,
            api_key=api_key,
            reasoning_effort=effective_reasoning_effort if llm == "api" else None,
        )
        typer.echo("Answer is ready.", err=True)
    except Exception as exc:
        report_error(logger, "query_generation_failed", "Query failed", error=str(exc))
        raise typer.Exit(code=1) from exc

    payload = (
        json.dumps(answer.model_dump(mode="json"), ensure_ascii=False)
        if effective_output_format == "json"
        else _render_query_text(question, answer)
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    else:
        typer.echo(payload)


@app.command()
def search(
    question: str = typer.Argument(..., metavar="QUESTION", help="The query string"),
    top_k: int | None = typer.Option(None, "--top-k", help="Number of results"),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="dense | sparse | hybrid"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Enable reranking"),
    query_explanation: bool = typer.Option(
        True,
        "--query-explanation/--no-query-explanation",
        help="Enable retrieval-oriented query expansion before retrieval",
    ),
    output_format: str = typer.Option("json", "--output-format", help="json | table"),
) -> None:
    settings = get_settings()
    effective_top_k = _resolve_retrieval_top_k(top_k)
    effective_mode = retrieval_mode or settings.retrieval_mode
    query_explanation_config = (
        QueryExplanationConfig(
            enabled=True,
            llm_model=settings.api_model,
            api_key=settings.api_key,
            base_url=settings.api_base_url,
            reasoning_effort=normalize_reasoning_effort(settings.query_explanation_reasoning_effort),
        )
        if query_explanation
        else None
    )
    if output_format not in {"json", "table"}:
        report_error(logger, "search_invalid_output_format", "Invalid output format", output_format=output_format)
        raise typer.Exit(code=2)
    if query_explanation:
        try:
            _ensure_query_explanation_api_configured()
        except typer.BadParameter as exc:
            typer.echo(str(exc), err=True)
            report_error(logger, "search_query_explanation_setup_failed", "Search setup failed", error=str(exc))
            raise typer.Exit(code=2) from exc

    try:
        typer.echo("Running retrieval...", err=True)
        start = time.perf_counter()
        hits = _retrieve_results(
            question,
            effective_top_k,
            effective_mode,
            rerank=rerank,
            query_explanation=query_explanation_config,
        )
        retrieval_latency_ms = (time.perf_counter() - start) * 1000
        typer.echo("Search results are ready.", err=True)
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
    top_k: int | None = typer.Option(None, "--top-k"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank"),
    query_explanation: bool = typer.Option(
        True,
        "--query-explanation/--no-query-explanation",
        help="Enable retrieval-oriented query expansion before retrieval",
    ),
    output_dir: Path = typer.Option(Path("data/benchmark_results/"), "--output-dir", help="Directory for report output"),
    tag: str = typer.Option("run", "--tag", help="Human-readable tag for this run"),
) -> None:
    effective_top_k = _resolve_retrieval_top_k(top_k)
    if query_explanation:
        try:
            _ensure_query_explanation_api_configured()
        except typer.BadParameter as exc:
            typer.echo(str(exc), err=True)
            report_error(logger, "benchmark_query_explanation_setup_failed", "Benchmark setup failed", error=str(exc))
            raise typer.Exit(code=2) from exc

    evaluator = Evaluator(
        dataset=dataset,
        retrieval_mode=retrieval_mode,
        top_k=effective_top_k,
        rerank=rerank,
        query_explanation=query_explanation,
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
