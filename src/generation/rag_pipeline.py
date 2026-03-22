"""RAG pipeline orchestration for retrieval, generation, and citation enrichment."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.config.settings import get_settings
from src.core.constants import FALLBACK_ANSWER
from src.core.errors import GenerationError
from src.core.logging import get_logger, report_error
from src.core.schemas import Citation, ChunkType, RAGAnswer
from src.generation.llm_client import (
    create_llm_client,
    fallback_generation_payload,
    normalize_generation_payload,
    parse_json_response,
)
from src.generation.prompt_templates import build_prompt
from src.generation.verifier import Verifier
from src.retrieval import get_retriever
from src.retrieval.reranker import Reranker

logger = get_logger(__name__)


def _to_str_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if item is not None]
        return " | ".join(part for part in parts if part)
    return str(value).strip()

def _normalize_chunk_id(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _context_by_chunk_id(contexts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(context.get("chunk_id", "")).strip(): context
        for context in contexts
        if str(context.get("chunk_id", "")).strip()
    }


def _backfill_citations(contexts: list[dict[str, Any]], max_items: int = 3) -> list[Citation]:
    citations: list[Citation] = []
    for context in contexts[:max_items]:
        chunk_id = str(context.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        evidence_text = str(context.get("text", "")).strip()
        if not evidence_text:
            continue
        citations.append(
            Citation(
                doc_id=str(context.get("doc_id", "")).strip(),
                chunk_id=chunk_id,
                page_number=context.get("page_number"),
                evidence_text=evidence_text[:300],
                evidence_type=ChunkType(str(context.get("chunk_type", "text") or "text")),
                headings=list(context.get("headings", []) or []),
                caption=str(context.get("caption", "")),
                asset_path=str(context.get("asset_path", "")),
            )
        )
    return citations


def _enrich_citations(raw_citations: Any, contexts: list[dict[str, Any]]) -> list[Citation]:
    context_map = _context_by_chunk_id(contexts)
    if not isinstance(raw_citations, list):
        return _backfill_citations(contexts)

    citations: list[Citation] = []
    for item in raw_citations:
        if not isinstance(item, dict):
            continue
        chunk_id = _normalize_chunk_id(item.get("chunk_id"))
        if not chunk_id or chunk_id not in context_map:
            continue

        context = context_map[chunk_id]
        evidence_text = _to_str_field(item.get("evidence_text")) or str(context.get("text", "")).strip()[:300]
        if not evidence_text:
            continue

        citations.append(
            Citation(
                doc_id=str(context.get("doc_id", "")).strip(),
                chunk_id=chunk_id,
                page_number=context.get("page_number"),
                evidence_text=evidence_text[:300],
                evidence_type=ChunkType(str(context.get("chunk_type", "text") or "text")),
                headings=list(context.get("headings", []) or []),
                caption=str(context.get("caption", "")),
                asset_path=str(context.get("asset_path", "")),
            )
        )

    return citations or _backfill_citations(contexts)


def _normalize_answer_payload(raw: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    payload = normalize_generation_payload(raw)
    answer_text = str(payload.get("answer", "") or "").strip()

    if not answer_text or answer_text == FALLBACK_ANSWER:
        return {
            "answer": FALLBACK_ANSWER,
            "supporting_materials": "is_blank",
            "explanation": "is_blank",
            "citations": [],
        }

    citations = _enrich_citations(payload.get("citations"), contexts)

    supporting_materials = _to_str_field(payload.get("supporting_materials"))
    if supporting_materials == "is_blank":
        supporting_materials = ""
    if not supporting_materials and citations:
        supporting_materials = citations[0].evidence_text

    explanation = _to_str_field(payload.get("explanation"))
    if explanation == "is_blank":
        explanation = ""
    if not explanation and citations:
        explanation = "Answer grounded in the cited evidence spans."

    return {
        "answer": answer_text,
        "supporting_materials": supporting_materials or "is_blank",
        "explanation": explanation or "is_blank",
        "citations": citations,
    }


def _retrieve_results(question: str, top_k: int, retrieval_mode: str, rerank: bool) -> list[dict[str, Any]]:
    retriever = get_retriever(retrieval_mode)
    fetch_k = top_k * 3 if rerank else top_k
    results = retriever.retrieve(question, fetch_k)
    if not rerank:
        return results[:top_k]
    reranker = Reranker()
    return reranker.rerank(question, results, top_k)


def _build_contexts(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for hit in hits:
        contexts.append(
            {
                "doc_id": str(hit.get("doc_id", "")).strip(),
                "chunk_id": str(hit.get("chunk_id", "")).strip(),
                "chunk_type": str(hit.get("chunk_type", "text") or "text"),
                "text": str(hit.get("text", "")).strip(),
                "page_number": hit.get("page_number"),
                "headings": list(hit.get("headings", []) or []),
                "caption": str(hit.get("caption", "") or ""),
                "asset_path": str(hit.get("asset_path", "") or ""),
            }
        )
    return contexts


def _parse_error_type(backend: str) -> str:
    return "generation_parse_error" if backend == "api" else "ollama_parse_error"


def _generate_payload_with_retry(
    *,
    client: Any,
    prompt: str,
    backend: str,
) -> dict[str, Any]:
    settings = get_settings()
    max_attempts = max(1, settings.request_retry_attempts)
    retry_delay = max(0.0, settings.request_retry_delay_seconds)

    for attempt in range(1, max_attempts + 1):
        try:
            raw_text = client.generate(prompt)
            payload = parse_json_response(raw_text)
            if payload is None:
                raise GenerationError(
                    error_type=_parse_error_type(backend),
                    message="Generation response was not valid JSON",
                    retryable=True,
                    context={"preview": raw_text[:300]},
                )
            return payload
        except GenerationError as exc:
            report_error(
                logger,
                "generation_failed",
                f"Generation failed ({exc.error_type})",
                llm_backend=backend,
                error_type=exc.error_type,
                attempt=attempt,
                detail=str(exc),
            )
            if not exc.retryable or attempt >= max_attempts:
                break
            time.sleep(retry_delay)

    return fallback_generation_payload()


@dataclass
class RAGConfig:
    top_k: int | None = None
    retrieval_mode: str | None = None
    rerank: bool | None = None
    llm_backend: str | None = None
    llm_model: str | None = None
    api_key: str | None = None

    @classmethod
    def from_settings(cls) -> "RAGConfig":
        settings = get_settings()
        return cls(
            top_k=settings.retrieval_top_k,
            retrieval_mode=settings.retrieval_mode,
            rerank=True,
            llm_backend=settings.llm_backend,
            llm_model=settings.api_model if settings.llm_backend == "api" else None,
        )

    def __post_init__(self) -> None:
        settings = get_settings()
        if self.top_k is None:
            self.top_k = settings.retrieval_top_k
        if self.retrieval_mode is None:
            self.retrieval_mode = settings.retrieval_mode
        if self.rerank is None:
            self.rerank = True
        if self.llm_backend is None:
            self.llm_backend = settings.llm_backend
        if self.llm_model is None and self.llm_backend == "api":
            self.llm_model = settings.api_model


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig.from_settings()
        self.verifier = Verifier()

    def answer_question(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        rerank: bool | None = None,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        api_key: str | None = None,
    ) -> RAGAnswer:
        settings = get_settings()
        effective_top_k = top_k or self.config.top_k or 5
        effective_mode = retrieval_mode or self.config.retrieval_mode or "hybrid"
        effective_rerank = bool(self.config.rerank if rerank is None else rerank)
        effective_llm = llm_backend or self.config.llm_backend or "api"
        effective_model = (
            llm_model or self.config.llm_model or settings.api_model
            if effective_llm == "api"
            else None
        )
        effective_api_key = api_key if api_key is not None else self.config.api_key

        retrieval_start = time.perf_counter()
        hits = _retrieve_results(question, effective_top_k, effective_mode, effective_rerank)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        contexts = _build_contexts(hits)

        generation_start = time.perf_counter()
        if contexts:
            prompt = build_prompt(question, contexts, [context["chunk_id"] for context in contexts])
            client = create_llm_client(
                effective_llm,
                model=effective_model,
                api_key=effective_api_key,
            )
            raw_payload = _generate_payload_with_retry(
                client=client,
                prompt=prompt,
                backend=effective_llm,
            )
        else:
            raw_payload = fallback_generation_payload()
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        normalized = _normalize_answer_payload(raw_payload, contexts)
        answer = RAGAnswer(
            answer=normalized["answer"],
            supporting_materials=normalized["supporting_materials"],
            explanation=normalized["explanation"],
            citations=normalized["citations"],
            retrieval_latency_ms=round(retrieval_latency_ms, 3),
            generation_latency_ms=round(generation_latency_ms, 3),
            retrieval_mode=effective_mode,
            llm_backend=effective_llm,
            verification=None,
        )

        verification = self.verifier.verify(answer, contexts)
        if verification.corrected_output:
            corrected_payload = answer.model_dump()
            corrected_payload.update(verification.corrected_output)
            answer = RAGAnswer(**corrected_payload, verification=verification)
        else:
            answer = answer.model_copy(update={"verification": verification})

        return answer
