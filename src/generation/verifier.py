from __future__ import annotations

import re
from typing import Any

from src.core.constants import FALLBACK_ANSWER
from src.core.schemas import Citation, RAGAnswer, VerificationResult

WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()


def _answer_payload(answer: RAGAnswer) -> dict[str, Any]:
    return {
        "answer": answer.answer,
        "answer_value": answer.answer_value,
        "answer_unit": answer.answer_unit,
        "ref_id": list(answer.ref_id),
        "supporting_materials": answer.supporting_materials,
        "explanation": answer.explanation,
        "citations": [citation.model_dump() for citation in answer.citations],
    }


def _fallback_payload() -> dict[str, Any]:
    return {
        "answer": FALLBACK_ANSWER,
        "answer_value": "is_blank",
        "answer_unit": "is_blank",
        "ref_id": [],
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
            warnings.append("No retrieved context above threshold.")
            fatal_issues += 1
        elif rag_answer.answer == FALLBACK_ANSWER:
            warnings.append("Generator returned fallback answer.")

        supported_citations = self._supported_citations(rag_answer.citations, contexts)
        if rag_answer.answer != FALLBACK_ANSWER and not supported_citations:
            warnings.append("No supported citations grounded in retrieved evidence.")
            fatal_issues += 1

        inferred_ref_ids = self._infer_ref_ids(supported_citations, contexts)
        if inferred_ref_ids and inferred_ref_ids != list(rag_answer.ref_id):
            corrected_output = _answer_payload(rag_answer)
            corrected_output["ref_id"] = inferred_ref_ids
            warnings.append("Reference IDs corrected from grounded citations.")

        if rag_answer.answer != FALLBACK_ANSWER and rag_answer.supporting_materials == "is_blank" and supported_citations:
            corrected_output = corrected_output or _answer_payload(rag_answer)
            corrected_output["supporting_materials"] = supported_citations[0].evidence_text
            warnings.append("Supporting materials backfilled from citations.")

        if rag_answer.answer != FALLBACK_ANSWER and not self._answer_value_supported(rag_answer, supported_citations, contexts):
            warnings.append("Answer value was not found verbatim in the cited evidence.")

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

        normalized_contexts = [_normalize_text(str(context.get("text", ""))) for context in contexts]
        supported: list[Citation] = []
        for citation in citations:
            evidence_text = _normalize_text(citation.evidence_text)
            if not evidence_text:
                continue
            if any(evidence_text in context_text for context_text in normalized_contexts):
                supported.append(citation)
        return supported

    def _infer_ref_ids(self, citations: list[Citation], contexts: list[dict[str, Any]]) -> list[str]:
        inferred = [citation.ref_id for citation in citations if citation.ref_id]
        if not inferred:
            inferred = [str(context.get("ref_id", "")) for context in contexts if context.get("ref_id")]
        return list(dict.fromkeys([ref_id for ref_id in inferred if ref_id]))[:3]

    def _answer_value_supported(
        self,
        answer: RAGAnswer,
        citations: list[Citation],
        contexts: list[dict[str, Any]],
    ) -> bool:
        value = str(answer.answer_value or "").strip()
        if not value or value == "is_blank":
            return True

        haystacks = [citation.evidence_text for citation in citations]
        if answer.supporting_materials and answer.supporting_materials != "is_blank":
            haystacks.append(answer.supporting_materials)
        haystacks.extend([str(context.get("text", "")) for context in contexts])

        normalized_value = _normalize_text(value)
        return any(normalized_value in _normalize_text(haystack) for haystack in haystacks if haystack)
