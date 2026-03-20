from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from fastapi import APIRouter, BackgroundTasks, status

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestAcceptedResponse,
    IngestRequest,
    PaperSummary,
    PapersResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.config.settings import get_settings
from src.core.logging import get_logger
from src.core.paths import find_project_root

logger = get_logger(__name__)
api_router = APIRouter(prefix="/api/v1")


class APIError(Exception):
    def __init__(self, *, status_code: int, code: str, message: str, detail: str = "") -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(message)


def _chunks_path(root: Path) -> Path:
    return root / "data" / "JSON" / "chunks.jsonl"


def _metadata_path(root: Path) -> Path:
    return root / "data" / "metadata" / "metadata.csv"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_chunk_counts(root: Path) -> tuple[dict[str, int], int]:
    path = _chunks_path(root)
    if not path.exists():
        return {}, 0

    counts: dict[str, int] = {}
    total_chunks = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            doc_id = str(payload.get("doc_id", "")).strip()
            if not doc_id:
                chunk_id = str(payload.get("chunk_id", "")).strip()
                doc_id = chunk_id.split("_", 1)[0] if "_" in chunk_id else chunk_id
            if not doc_id:
                continue

            counts[doc_id] = counts.get(doc_id, 0) + 1
            total_chunks += 1

    return counts, total_chunks


def _load_paper_metadata(root: Path) -> dict[str, dict[str, Any]]:
    path = _metadata_path(root)
    if not path.exists():
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = str(row.get("id", "")).strip()
            if not doc_id:
                continue
            metadata[doc_id] = {
                "title": str(row.get("title", "")).strip() or doc_id,
                "year": _safe_int(row.get("year"), 0),
            }
    return metadata


def _papers_payload() -> PapersResponse:
    root = find_project_root()
    chunk_counts, _ = _load_chunk_counts(root)
    metadata = _load_paper_metadata(root)

    doc_ids = sorted(set(metadata) | set(chunk_counts))
    papers = [
        PaperSummary(
            id=doc_id,
            title=str(metadata.get(doc_id, {}).get("title", doc_id)),
            year=_safe_int(metadata.get(doc_id, {}).get("year"), 0),
            num_chunks=chunk_counts.get(doc_id, 0),
        )
        for doc_id in doc_ids
    ]
    return PapersResponse(papers=papers, total=len(papers))


def _index_loaded() -> bool:
    from src.indexing.bm25_index import BM25Index
    from src.indexing.vector_store import FaissStore

    try:
        return FaissStore().load() and BM25Index().load()
    except Exception as exc:  # pragma: no cover - defensive path for corrupt artifacts
        logger.warning("api_index_health_check_failed", error=str(exc))
        return False


def _retrieve_results(question: str, top_k: int, retrieval_mode: str, rerank: bool) -> list[dict[str, Any]]:
    from src.retrieval import get_retriever
    from src.retrieval.reranker import Reranker

    retriever = get_retriever(retrieval_mode)
    fetch_k = top_k * 3 if rerank else top_k
    results = retriever.retrieve(question, fetch_k)
    if not rerank:
        return results[:top_k]

    reranker = Reranker()
    return reranker.rerank(question, results, top_k)


def _build_rag_pipeline(config: Any) -> Any:
    from src.generation.rag_pipeline import RAGPipeline

    return RAGPipeline(config=config)


def _is_index_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return isinstance(exc, FileNotFoundError) or any(
        marker in detail
        for marker in (
            "index not loaded",
            "embeddings file not found",
            "chunks file not found",
        )
    )


def _ingest_source_not_found(exc: Exception) -> bool:
    return isinstance(exc, FileNotFoundError)


def _normalize_request_path(path_value: str) -> Path | None:
    path_text = str(path_value or "").strip()
    return Path(path_text) if path_text else None


def _run_ingest_job(source: str, path_value: str) -> None:
    from src.main import ingest as ingest_command

    try:
        ingest_command(
            source=source,
            path=_normalize_request_path(path_value),
            workers=None,
            rebuild_index=False,
            retry_failed=False,
            dry_run=False,
        )
    except typer.Exit as exc:
        logger.info("api_ingest_finished", source=source, exit_code=exc.exit_code)
    except Exception as exc:  # pragma: no cover - background task failures are logged
        logger.error("api_ingest_failed", source=source, error=str(exc))


@api_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    papers = _papers_payload()
    _, num_chunks = _load_chunk_counts(find_project_root())
    return HealthResponse(
        status="ok",
        index_loaded=_index_loaded(),
        num_papers=papers.total,
        num_chunks=num_chunks,
        embedding_model=settings.embedding_model,
        retrieval_modes_available=["dense", "sparse", "hybrid"],
    )


@api_router.get("/papers", response_model=PapersResponse)
def papers() -> PapersResponse:
    return _papers_payload()


@api_router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def search(payload: SearchRequest) -> SearchResponse:
    settings = get_settings()
    retrieval_mode = payload.retrieval_mode or settings.retrieval_mode

    try:
        start = time.perf_counter()
        hits = _retrieve_results(payload.question, payload.top_k, retrieval_mode, rerank=payload.rerank)
        retrieval_latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        if _is_index_error(exc):
            raise APIError(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="INDEX_NOT_LOADED",
                message="Search index is not loaded.",
                detail=str(exc),
            ) from exc
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="RETRIEVAL_FAILED",
            message="Failed to retrieve results.",
            detail=str(exc),
        ) from exc

    results = [
        SearchResult(
            rank=int(hit.get("rank", index)),
            ref_id=str(hit.get("ref_id", "")),
            score=float(hit.get("score", 0.0)),
            text=str(hit.get("text", "")),
            page=hit.get("page"),
            source_file=str(hit.get("source_file", "")),
            headings=list(hit.get("headings", []) or []),
        )
        for index, hit in enumerate(hits, start=1)
    ]
    return SearchResponse(
        results=results,
        retrieval_latency_ms=round(retrieval_latency_ms, 3),
        retrieval_mode=retrieval_mode,
        total_results=len(results),
    )


@api_router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def query(payload: QueryRequest) -> QueryResponse:
    from src.generation.rag_pipeline import RAGConfig

    settings = get_settings()
    retrieval_mode = payload.retrieval_mode or settings.retrieval_mode
    llm_backend = payload.llm_backend or settings.llm_backend

    try:
        pipeline = _build_rag_pipeline(
            RAGConfig(
                top_k=payload.top_k,
                retrieval_mode=retrieval_mode,
                rerank=payload.rerank,
                llm_backend=llm_backend,
            )
        )
        return pipeline.answer_question(
            payload.question,
            top_k=payload.top_k,
            retrieval_mode=retrieval_mode,
            rerank=payload.rerank,
            llm_backend=llm_backend,
        )
    except Exception as exc:
        if _is_index_error(exc):
            raise APIError(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                code="INDEX_NOT_LOADED",
                message="Search index is not loaded.",
                detail=str(exc),
            ) from exc
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="GENERATION_FAILED",
            message="Failed to generate an answer.",
            detail=str(exc),
        ) from exc


@api_router.post(
    "/ingest",
    response_model=IngestAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def ingest(payload: IngestRequest, background_tasks: BackgroundTasks) -> IngestAcceptedResponse:
    from src.ingestion.sources import create_source

    path = _normalize_request_path(payload.path)

    try:
        source_impl = create_source(payload.source, path)
        documents = source_impl.discover()
    except Exception as exc:
        if _ingest_source_not_found(exc):
            raise APIError(
                status_code=status.HTTP_404_NOT_FOUND,
                code="SOURCE_NOT_FOUND",
                message="The requested ingestion source could not be found.",
                detail=str(exc),
            ) from exc
        if isinstance(exc, ValueError):
            raise APIError(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                code="VALIDATION_ERROR",
                message="The ingestion request is invalid.",
                detail=str(exc),
            ) from exc
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_ERROR",
            message="Failed to start ingestion.",
            detail=str(exc),
        ) from exc

    task_id = f"ingest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    background_tasks.add_task(_run_ingest_job, payload.source, payload.path)
    return IngestAcceptedResponse(
        status="accepted",
        message=f"Ingestion started for {len(documents)} documents",
        task_id=task_id,
    )
