from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, status

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PaperSummary,
    PapersResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from src.config.settings import get_settings, normalize_reasoning_effort
from src.core.logging import get_logger
from src.core.paths import chunks_jsonl_path, find_project_root
from src.core.schemas import SearchResult
from src.retrieval.query_explanation import QueryExplanationConfig, retrieve_with_optional_query_explanation

logger = get_logger(__name__)
api_router = APIRouter(prefix="/api/v1")


class APIError(Exception):
    def __init__(self, *, status_code: int, code: str, message: str, detail: str = "") -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(message)


def _load_chunk_counts(root: Path) -> tuple[dict[str, int], int]:
    path = chunks_jsonl_path(root)
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


def _papers_payload() -> PapersResponse:
    root = find_project_root()
    chunk_counts, _ = _load_chunk_counts(root)
    papers = [
        PaperSummary(doc_id=doc_id, num_chunks=chunk_counts[doc_id])
        for doc_id in sorted(chunk_counts)
    ]
    return PapersResponse(papers=papers, total=len(papers))


def _index_loaded() -> bool:
    from src.indexing.bm25_index import BM25Index
    from src.indexing.vector_store import FaissStore

    try:
        return FaissStore().load() and BM25Index().load()
    except Exception as exc:
        logger.warning("api_index_health_check_failed", error=str(exc))
        return False


def _retrieve_results(
    question: str,
    top_k: int,
    retrieval_mode: str,
    rerank: bool,
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
    query_explanation_config = (
        QueryExplanationConfig(
            enabled=True,
            llm_model=settings.api_model,
            api_key=settings.api_key,
            base_url=settings.api_base_url,
            reasoning_effort=normalize_reasoning_effort(settings.query_explanation_reasoning_effort),
        )
        if payload.query_explanation
        else None
    )

    try:
        start = time.perf_counter()
        hits = _retrieve_results(
            payload.question,
            payload.top_k,
            retrieval_mode,
            rerank=payload.rerank,
            query_explanation=query_explanation_config,
        )
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
                query_explanation_enabled=payload.query_explanation,
                llm_backend=llm_backend,
                llm_model=(payload.llm_model or settings.api_model) if llm_backend == "api" else None,
            )
        )
        return pipeline.answer_question(
            payload.question,
            top_k=payload.top_k,
            retrieval_mode=retrieval_mode,
            rerank=payload.rerank,
            query_explanation_enabled=payload.query_explanation,
            llm_backend=llm_backend,
            llm_model=(payload.llm_model or settings.api_model) if llm_backend == "api" else None,
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
