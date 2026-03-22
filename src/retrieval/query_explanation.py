from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.config.settings import get_settings
from src.core.constants import RRF_K
from src.core.errors import GenerationError
from src.core.logging import get_logger
from src.generation.llm_client import create_llm_client
from src.retrieval import get_retriever
from src.retrieval.reranker import Reranker

logger = get_logger(__name__)

QUERY_EXPLANATION_SYSTEM_PROMPT = "Return only one compact retrieval query in English."

QUERY_EXPLANATION_PROMPT = """You are rewriting academic questions into compact retrieval queries for a RAG system.

Task:
Rewrite the user's question into a short retrieval query that improves evidence recall from academic papers.

Requirements:
- Output English only.
- Return a single compact retrieval query only, not an explanation.
- Do not answer the question.
- Do not use outside knowledge.
- Preserve the original topic, named entities, constraints, technical terms, relationships, and any phase or stage qualifiers in the question.
- When the question targets a specific model, method, dataset, experiment, or paper entity, keep that target as the primary anchor of the query.
- Keep named entities, target methods, model names, dataset names, metrics, task targets, conditions, phase or stage constraints, and requested evidence focus central when they are present.
- Do not let generic calculation, definition, background, or conversion terms dominate the query.
- Expand only into retrieval-useful evidence fields, such as reported values, counts, baselines, comparison targets, metric definitions, experimental settings, setup or configuration details, hardware or resource details, phase-specific details, time conditions, denominators, or referenced evidence fields.
- Prefer evidence-bearing terms that are likely to appear in body text, section headers, figure captions, table text, footnotes, appendix descriptions, or reported results.
- For questions involving comparison, estimation, aggregation, or conversion, emphasize the quantities, counts, setup details, hardware details, phase-specific constraints, and evidence fields that must be retrieved, not the reasoning process itself.
- Prefer paper-specific evidence cues over generic explanatory wording.
- Avoid tutorial wording, broad background concepts, verbose paraphrasing, and unnecessary helper terms.
- Do not invent facts, numbers, definitions, or assumptions not present in the question.
- Keep the output concise and information-dense, while preserving the target entity, requested evidence focus, and key constraints.
- Keep the rewrite broadly applicable across academic domains.

Output goal:
Produce a compact retrieval query optimized for evidence lookup in academic papers.

User question:
{question}
"""


@dataclass
class RetrievalExecution:
    results: list[dict[str, Any]]
    expanded_query: str | None = None


@dataclass
class QueryExplanationConfig:
    enabled: bool = False
    llm_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None


def _normalize_expanded_query(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _fuse_ranked_result_sets(result_sets: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    fused: dict[str, dict[str, Any]] = {}
    for result_set in result_sets:
        for item in result_set:
            chunk_id = str(item.get("chunk_id", "")).strip()
            if not chunk_id:
                continue

            existing = fused.get(chunk_id)
            rrf_score = 1.0 / (RRF_K + int(item["rank"]))
            if existing is None:
                fused[chunk_id] = {**item, "score": rrf_score}
            else:
                existing["score"] += rrf_score

    ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    for index, item in enumerate(ordered, start=1):
        item["rank"] = index
    return ordered


class QueryExplainer:
    def __init__(
        self,
        *,
        llm_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.llm_model = llm_model
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort

    def expand(self, question: str) -> str:
        settings = get_settings()
        client = create_llm_client(
            "api",
            model=self.llm_model or settings.api_model,
            api_key=self.api_key if self.api_key is not None else settings.api_key,
            base_url=self.base_url if self.base_url is not None else settings.api_base_url,
            reasoning_effort=self.reasoning_effort,
        )
        prompt = QUERY_EXPLANATION_PROMPT.format(question=question.strip())

        max_attempts = max(1, settings.request_retry_attempts)
        retry_delay = max(0.0, settings.request_retry_delay_seconds)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                expanded_query = _normalize_expanded_query(
                    client.generate(prompt, system_prompt=QUERY_EXPLANATION_SYSTEM_PROMPT)
                )
                if not expanded_query:
                    raise GenerationError(
                        error_type="query_explanation_empty_output",
                        message="Query explanation returned empty output",
                        retryable=True,
                    )
                return expanded_query
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "query_explanation_attempt_failed",
                    attempt=attempt,
                    error_type=getattr(exc, "error_type", exc.__class__.__name__),
                    detail=str(exc),
                )
                retryable = bool(getattr(exc, "retryable", True))
                if not retryable or attempt >= max_attempts:
                    break
                time.sleep(retry_delay)

        raise GenerationError(
            error_type="query_explanation_failed",
            message=str(last_error or "Query explanation failed"),
            retryable=False,
        )


def retrieve_with_optional_query_explanation(
    question: str,
    *,
    top_k: int,
    retrieval_mode: str,
    rerank: bool,
    query_explanation: QueryExplanationConfig | None = None,
) -> RetrievalExecution:
    retriever = get_retriever(retrieval_mode)
    fetch_k = top_k * 5 if rerank else top_k

    original_results = retriever.retrieve(question, fetch_k)
    explanation_config = query_explanation or QueryExplanationConfig()
    if not explanation_config.enabled:
        if not rerank:
            return RetrievalExecution(results=original_results[:top_k])
        reranker = Reranker()
        return RetrievalExecution(results=reranker.rerank(question, original_results, top_k))

    try:
        expanded_query = QueryExplainer(
            llm_model=explanation_config.llm_model,
            api_key=explanation_config.api_key,
            base_url=explanation_config.base_url,
            reasoning_effort=explanation_config.reasoning_effort,
        ).expand(question)
    except GenerationError as exc:
        logger.warning(
            "query_explanation_fallback_to_original_query",
            error_type=exc.error_type,
            detail=str(exc),
        )
        expanded_query = ""

    if not expanded_query or expanded_query == question.strip():
        if not rerank:
            return RetrievalExecution(results=original_results[:top_k])
        reranker = Reranker()
        return RetrievalExecution(results=reranker.rerank(question, original_results, top_k))

    expanded_results = retriever.retrieve(expanded_query, fetch_k)
    merged_candidates = _fuse_ranked_result_sets([original_results, expanded_results], fetch_k)
    if not rerank:
        return RetrievalExecution(results=merged_candidates[:top_k], expanded_query=expanded_query)

    reranker = Reranker()
    reranked_results = reranker.rerank(expanded_query, merged_candidates, top_k)
    return RetrievalExecution(results=reranked_results, expanded_query=expanded_query)
