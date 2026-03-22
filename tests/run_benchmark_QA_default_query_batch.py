from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from src.config.settings import get_settings, normalize_reasoning_effort
from src.core.schemas import RAGAnswer
from src.evaluation.metrics import mrr, ndcg_at_k, recall_at_k
from src.generation.llm_client import create_llm_client, fallback_generation_payload
from src.generation.prompt_templates import build_prompt
from src.generation.rag_pipeline import RAGPipeline, _build_contexts, _generate_payload_with_retry, _normalize_answer_payload
from src.retrieval.query_explanation import QueryExplanationConfig, RetrievalExecution, retrieve_with_optional_query_explanation

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "benchmark_QA.csv"
OUTPUT_CSV_PATH = REPO_ROOT / "tests" / "benchmark_QA_default_query_results.csv"


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_ref_doc_ids(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text in {"N/A", "is_blank"}:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _default_query_explanation_config(pipeline: RAGPipeline) -> QueryExplanationConfig:
    settings = get_settings()
    effective_llm = pipeline.config.llm_backend or settings.llm_backend or "api"
    effective_model = pipeline.config.llm_model or settings.api_model if effective_llm == "api" else None
    effective_api_key = pipeline.config.api_key if pipeline.config.api_key is not None else settings.api_key
    return QueryExplanationConfig(
        enabled=bool(pipeline.config.query_explanation_enabled),
        llm_model=effective_model if effective_llm == "api" else settings.api_model,
        api_key=effective_api_key if effective_llm == "api" else settings.api_key,
        base_url=settings.api_base_url,
        reasoning_effort=normalize_reasoning_effort(settings.query_explanation_reasoning_effort),
    )


def _default_retrieval_execution(pipeline: RAGPipeline, question: str) -> RetrievalExecution:
    settings = get_settings()
    return retrieve_with_optional_query_explanation(
        question,
        top_k=int(pipeline.config.top_k or settings.retrieval_top_k),
        retrieval_mode=str(pipeline.config.retrieval_mode or settings.retrieval_mode),
        rerank=bool(pipeline.config.rerank),
        query_explanation=_default_query_explanation_config(pipeline),
    )


def _answer_from_execution(
    pipeline: RAGPipeline,
    question: str,
    execution: RetrievalExecution,
    retrieval_latency_ms: float,
) -> RAGAnswer:
    settings = get_settings()
    effective_mode = str(pipeline.config.retrieval_mode or settings.retrieval_mode)
    effective_llm = str(pipeline.config.llm_backend or settings.llm_backend or "api")
    effective_model = (
        pipeline.config.llm_model or settings.api_model
        if effective_llm == "api"
        else None
    )
    effective_api_key = pipeline.config.api_key if pipeline.config.api_key is not None else settings.api_key
    effective_reasoning_effort = pipeline.config.reasoning_effort

    contexts = _build_contexts(execution.results)

    generation_start = time.perf_counter()
    if contexts:
        prompt = build_prompt(question, contexts, [context["chunk_id"] for context in contexts])
        client = create_llm_client(
            effective_llm,
            model=effective_model,
            api_key=effective_api_key,
            reasoning_effort=effective_reasoning_effort,
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

    verification = pipeline.verifier.verify(answer, contexts)
    if verification.corrected_output:
        corrected_payload = answer.model_dump()
        corrected_payload.update(verification.corrected_output)
        return RAGAnswer(**corrected_payload, verification=verification)
    return answer.model_copy(update={"verification": verification})


def _write_results_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "question",
        "expected_answer",
        "expected_supporting_materials",
        "expected_explanation",
        "top_k",
        "retrieval_mode",
        "rerank",
        "query_explanation_enabled",
        "llm_backend",
        "expanded_query",
        "relevant_ref_doc_ids",
        "retrieved_doc_ids",
        "retrieved_chunk_ids",
        "retrieved_ref_ids",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "answer",
        "supporting_materials",
        "explanation",
        "citation_doc_ids",
        "citation_chunk_ids",
        "verification_passed",
        "verification_confidence",
        "verification_warnings",
        "error_type",
        "error_message",
    ]

    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    pipeline = RAGPipeline()
    settings = get_settings()
    effective_top_k = int(pipeline.config.top_k or settings.retrieval_top_k)
    effective_mode = str(pipeline.config.retrieval_mode or settings.retrieval_mode)
    effective_rerank = bool(pipeline.config.rerank)
    effective_query_explanation = bool(pipeline.config.query_explanation_enabled)
    effective_llm_backend = str(pipeline.config.llm_backend or settings.llm_backend or "api")

    rows: list[dict[str, Any]] = []

    with DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            question_id = str(raw.get("question_id", raw.get("id", ""))).strip()
            question = str(raw.get("question", "")).strip()
            if not question:
                continue
            relevant_ref_doc_ids = _parse_ref_doc_ids(raw.get("ref_doc_id", raw.get("ref_id")))

            row: dict[str, Any] = {
                "question_id": question_id,
                "question": question,
                "expected_answer": str(raw.get("answer", "")).strip(),
                "expected_supporting_materials": str(raw.get("supporting_materials", "")).strip(),
                "expected_explanation": str(raw.get("explanation", "")).strip(),
                "top_k": effective_top_k,
                "retrieval_mode": effective_mode,
                "rerank": effective_rerank,
                "query_explanation_enabled": effective_query_explanation,
                "llm_backend": effective_llm_backend,
                "expanded_query": "",
                "relevant_ref_doc_ids": _json_cell(relevant_ref_doc_ids),
                "retrieved_doc_ids": _json_cell([]),
                "retrieved_chunk_ids": _json_cell([]),
                "retrieved_ref_ids": _json_cell([]),
                "recall_at_k": "",
                "mrr": "",
                "ndcg_at_k": "",
                "retrieval_latency_ms": "",
                "generation_latency_ms": "",
                "answer": "",
                "supporting_materials": "",
                "explanation": "",
                "citation_doc_ids": _json_cell([]),
                "citation_chunk_ids": _json_cell([]),
                "verification_passed": "",
                "verification_confidence": "",
                "verification_warnings": _json_cell([]),
                "error_type": "",
                "error_message": "",
            }

            try:
                retrieval_start = time.perf_counter()
                execution = _default_retrieval_execution(pipeline, question)
                retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
                answer = _answer_from_execution(pipeline, question, execution, retrieval_latency_ms)

                retrieved_doc_ids = [str(item.get("doc_id", "")).strip() for item in execution.results if item.get("doc_id")]
                retrieved_chunk_ids = [str(item.get("chunk_id", "")).strip() for item in execution.results if item.get("chunk_id")]
                citation_doc_ids = [citation.doc_id for citation in answer.citations]
                citation_chunk_ids = [citation.chunk_id for citation in answer.citations]
                verification_warnings = answer.verification.warnings if answer.verification else []

                row.update(
                    {
                        "expanded_query": execution.expanded_query or "",
                        "retrieved_doc_ids": _json_cell(retrieved_doc_ids),
                        "retrieved_chunk_ids": _json_cell(retrieved_chunk_ids),
                        "retrieved_ref_ids": _json_cell(retrieved_doc_ids),
                        "recall_at_k": recall_at_k(retrieved_doc_ids, relevant_ref_doc_ids, effective_top_k),
                        "mrr": mrr(retrieved_doc_ids, relevant_ref_doc_ids),
                        "ndcg_at_k": ndcg_at_k(retrieved_doc_ids, relevant_ref_doc_ids, effective_top_k),
                        "retrieval_latency_ms": answer.retrieval_latency_ms,
                        "generation_latency_ms": answer.generation_latency_ms,
                        "answer": answer.answer,
                        "supporting_materials": answer.supporting_materials,
                        "explanation": answer.explanation,
                        "citation_doc_ids": _json_cell(citation_doc_ids),
                        "citation_chunk_ids": _json_cell(citation_chunk_ids),
                        "verification_passed": answer.verification.passed if answer.verification else "",
                        "verification_confidence": answer.verification.confidence if answer.verification else "",
                        "verification_warnings": _json_cell(verification_warnings),
                    }
                )
            except Exception as exc:
                row["error_type"] = exc.__class__.__name__
                row["error_message"] = str(exc)

            rows.append(row)
            print(f"[{question_id}] done")

    _write_results_csv(rows)
    error_count = sum(1 for row in rows if row["error_type"])
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV_PATH}")
    print(f"Rows with errors: {error_count}")


if __name__ == "__main__":
    main()
