from __future__ import annotations

import re
from typing import Any

from src.core.constants import FALLBACK_ANSWER
from src.core.schemas import Citation, RAGAnswer, VerificationResult

WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()


def _answer_payload(answer: RAGAnswer) -> dict[str, Any]:
    return answer.model_dump(mode="json")


def _fallback_payload() -> dict[str, Any]:
    return {
        "answer": FALLBACK_ANSWER,
        "supporting_materials": "is_blank",
        "explanation": "is_blank",
        "citations": [],
    }


class Verifier:
    def verify(self, answer: RAGAnswer | dict[str, Any], contexts: list[dict[str, Any]] | None = None) -> VerificationResult:
        rag_answer = answer if isinstance(answer, RAGAnswer) else RAGAnswer(**answer)
        contexts = contexts or []
        warnings: list[str] = []
        corrected_output: dict[str, Any] | None = None
        fatal_issues = 0

        if not contexts:
            warnings.append("No retrieved context available.")
            fatal_issues += 1
        elif rag_answer.answer == FALLBACK_ANSWER:
            warnings.append("Generator returned fallback answer.")

        supported_citations = self._supported_citations(rag_answer.citations, contexts)
        if rag_answer.answer != FALLBACK_ANSWER and not supported_citations:
            warnings.append("No supported citations grounded in retrieved evidence.")
            fatal_issues += 1

        if rag_answer.answer != FALLBACK_ANSWER and rag_answer.supporting_materials == "is_blank" and supported_citations:
            corrected_output = _answer_payload(rag_answer)
            corrected_output["supporting_materials"] = supported_citations[0].evidence_text
            warnings.append("Supporting materials backfilled from citations.")

        if fatal_issues > 0:
            corrected_output = _fallback_payload()

        confidence = 1.0
        confidence -= 0.45 * fatal_issues
        confidence -= 0.1 * max(0, len(warnings) - fatal_issues)
        confidence = max(0.0, min(1.0, confidence))

        return VerificationResult(
            passed=fatal_issues == 0,
            confidence=round(confidence, 3),
            warnings=warnings,
            corrected_output=corrected_output,
        )

    def _supported_citations(self, citations: list[Citation], contexts: list[dict[str, Any]]) -> list[Citation]:
        if not citations:
            return []

        context_by_chunk_id = {
            str(context.get("chunk_id", "")).strip(): _normalize_text(str(context.get("text", "")))
            for context in contexts
            if context.get("chunk_id")
        }
        supported: list[Citation] = []
        for citation in citations:
            chunk_id = citation.chunk_id.strip()
            if not chunk_id or chunk_id not in context_by_chunk_id:
                continue
            evidence_text = _normalize_text(citation.evidence_text)
            if evidence_text and evidence_text in context_by_chunk_id[chunk_id]:
                supported.append(citation)
        return supported
