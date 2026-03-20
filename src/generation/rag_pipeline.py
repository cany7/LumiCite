"""RAG pipeline orchestration for retrieval, generation, and normalization."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.core.constants import FALLBACK_ANSWER
from src.core.logging import get_logger
from src.core.paths import find_project_root
from src.core.schemas import Citation, RAGAnswer
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
        parts = [str(x).strip() for x in value if x is not None]
        return " | ".join(parts[:3])
    return str(value).strip()


def _format_list_py(items: list[str]) -> str:
    safe = [str(x).strip() for x in items if x]
    inner = ",".join([f"'{x}'" for x in safe])
    return f"[{inner}]"


def _load_metadata_map(root: Path) -> dict[str, dict[str, str]]:
    import csv

    meta_path = root / "data" / "metadata" / "metadata.csv"
    metadata: dict[str, dict[str, str]] = {}
    if not meta_path.exists():
        return metadata

    with meta_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ref_id = row.get("id")
            if not ref_id:
                continue
            metadata[ref_id] = {
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "citation": row.get("citation", ""),
            }
    return metadata


def _normalize_ref_id(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    return text.split("_", 1)[0] if "_" in text else text


def _backfill_citations(contexts: list[dict[str, Any]], max_items: int = 3) -> list[Citation]:
    citations: list[Citation] = []
    for context in contexts[:max_items]:
        evidence_text = str(context.get("text", "")).strip()
        if not evidence_text:
            continue
        citations.append(
            Citation(
                ref_id=str(context.get("ref_id", "")),
                page=context.get("page"),
                evidence_text=evidence_text[:300],
                evidence_type=str(context.get("chunk_type", "text") or "text"),
            )
        )
    return citations


def _normalize_citations(raw_citations: Any, contexts: list[dict[str, Any]]) -> list[Citation]:
    if not isinstance(raw_citations, list):
        return _backfill_citations(contexts)

    allowed_ref_ids = {str(context.get("ref_id", "")) for context in contexts if context.get("ref_id")}
    normalized: list[Citation] = []
    for item in raw_citations:
        if not isinstance(item, dict):
            continue
        ref_id = _normalize_ref_id(item.get("ref_id"))
        if not ref_id or (allowed_ref_ids and ref_id not in allowed_ref_ids):
            continue

        page = item.get("page")
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None

        evidence_text = str(item.get("evidence_text", "")).strip()
        if not evidence_text:
            context = next((entry for entry in contexts if entry.get("ref_id") == ref_id), None)
            evidence_text = str(context.get("text", "")).strip()[:300] if context else ""
        if not evidence_text:
            continue

        evidence_type = str(item.get("evidence_type", "text") or "text")
        normalized.append(
            Citation(
                ref_id=ref_id,
                page=page,
                evidence_text=evidence_text[:300],
                evidence_type=evidence_type,
            )
        )

    return normalized or _backfill_citations(contexts)


def _normalize_answer_payload(raw: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    payload = normalize_generation_payload(raw)
    answer_text = str(payload.get("answer", "") or "").strip()

    if not answer_text or answer_text == FALLBACK_ANSWER:
        return {
            "answer": FALLBACK_ANSWER,
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": [],
            "supporting_materials": "is_blank",
            "explanation": "is_blank",
            "citations": [],
        }

    citations = _normalize_citations(payload.get("citations"), contexts)

    ref_ids = payload.get("ref_id", [])
    if isinstance(ref_ids, str):
        ref_ids = [ref_ids]
    if not isinstance(ref_ids, list):
        ref_ids = []

    normalized_ref_ids = [_normalize_ref_id(ref_id) for ref_id in ref_ids if ref_id]
    if not normalized_ref_ids:
        normalized_ref_ids = [citation.ref_id for citation in citations]
    if not normalized_ref_ids:
        normalized_ref_ids = [str(context.get("ref_id", "")) for context in contexts if context.get("ref_id")]

    normalized_ref_ids = list(dict.fromkeys([ref_id for ref_id in normalized_ref_ids if ref_id]))[:3]
    supporting_materials = _to_str_field(payload.get("supporting_materials"))
    if not supporting_materials and citations:
        supporting_materials = citations[0].evidence_text

    explanation = _to_str_field(payload.get("explanation"))
    if not explanation and citations:
        explanation = "Answer grounded in the cited evidence spans."

    answer_value = payload.get("answer_value", "is_blank")
    if isinstance(answer_value, list):
        answer_value = json.dumps(answer_value, ensure_ascii=False)
    if answer_value in (None, ""):
        answer_value = answer_text

    answer_unit = str(payload.get("answer_unit", "is_blank") or "is_blank")
    if answer_text.upper() in {"TRUE", "FALSE"}:
        answer_value = "1" if answer_text.upper() == "TRUE" else "0"
        answer_unit = "is_blank"

    return {
        "answer": answer_text,
        "answer_value": str(answer_value),
        "answer_unit": answer_unit,
        "ref_id": normalized_ref_ids,
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


def _build_contexts(hits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    contexts: list[dict[str, Any]] = []
    candidate_ref_ids: list[str] = []
    for hit in hits:
        ref_id = _normalize_ref_id(hit.get("ref_id"))
        if ref_id:
            candidate_ref_ids.append(ref_id)
        contexts.append(
            {
                "ref_id": ref_id,
                "text": str(hit.get("text", "")),
                "page": hit.get("page"),
                "headings": list(hit.get("headings", []) or []),
                "chunk_type": str(hit.get("chunk_type", "text") or "text"),
            }
        )

    deduped_ref_ids = list(dict.fromkeys([ref_id for ref_id in candidate_ref_ids if ref_id]))
    return contexts, deduped_ref_ids


@dataclass
class RAGConfig:
    top_k: int | None = None
    retrieval_mode: str | None = None
    rerank: bool | None = None
    llm_backend: str | None = None

    @classmethod
    def from_settings(cls) -> "RAGConfig":
        settings = get_settings()
        return cls(
            top_k=settings.retrieval_top_k,
            retrieval_mode=settings.retrieval_mode,
            rerank=True,
            llm_backend=settings.llm_backend,
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


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig.from_settings()
        self.root = find_project_root()
        self.meta_map = _load_metadata_map(self.root)
        self.verifier = Verifier()

    def answer_question(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        rerank: bool | None = None,
        llm_backend: str | None = None,
    ) -> RAGAnswer:
        effective_top_k = top_k or self.config.top_k or 5
        effective_mode = retrieval_mode or self.config.retrieval_mode or "hybrid"
        effective_rerank = self.config.rerank if rerank is None else rerank
        effective_llm = llm_backend or self.config.llm_backend or "gemini"

        retrieval_start = time.perf_counter()
        hits = _retrieve_results(question, effective_top_k, effective_mode, effective_rerank)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        contexts, candidate_ref_ids = _build_contexts(hits)

        generation_start = time.perf_counter()
        if contexts:
            prompt = build_prompt(question, contexts, candidate_ref_ids)
            client = create_llm_client(effective_llm)
            raw_text = client.generate(prompt)
            raw_payload = parse_json_response(raw_text) or fallback_generation_payload()
        else:
            raw_payload = fallback_generation_payload()
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        normalized = _normalize_answer_payload(raw_payload, contexts)
        answer = RAGAnswer(
            answer=normalized["answer"],
            answer_value=normalized["answer_value"],
            answer_unit=normalized["answer_unit"],
            ref_id=normalized["ref_id"],
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

    def answer(self, qid: str, question: str) -> dict[str, Any]:
        answer = self.answer_question(question)

        ref_urls = [self.meta_map.get(ref_id, {}).get("url", "") for ref_id in answer.ref_id]
        ref_id_str = _format_list_py(answer.ref_id) if answer.ref_id else "is_blank"
        ref_url_str = _format_list_py([url for url in ref_urls if url]) if any(ref_urls) else "is_blank"

        return {
            "id": qid,
            "question": question,
            "answer": answer.answer,
            "answer_value": answer.answer_value,
            "answer_unit": answer.answer_unit,
            "ref_id": ref_id_str,
            "ref_url": ref_url_str,
            "supporting_materials": answer.supporting_materials,
            "explanation": answer.explanation,
        }
