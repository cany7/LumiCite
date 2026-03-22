from __future__ import annotations

import argparse
import json
import sys
import time

from src.config.settings import get_settings
from src.core.schemas import RAGAnswer
from src.generation.llm_client import create_llm_client, fallback_generation_payload
from src.generation.prompt_templates import build_prompt
from src.generation.rag_pipeline import (
    RAGConfig,
    RAGPipeline,
    _build_contexts,
    _generate_payload_with_retry,
    _normalize_answer_payload,
    _retrieve_results,
)


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Debug generation using the current retrieval + rerank + prompt + generation pipeline."
    )
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k, help="Final context count")
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "sparse", "hybrid"),
        default=settings.retrieval_mode,
        help="Retriever mode",
    )
    parser.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=True,
        help="Enable reranking",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable reranking",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("api", "ollama"),
        default=settings.llm_backend,
        help="Generation backend",
    )
    parser.add_argument("--model", default=None, help="Generation model override for api backend")
    args = parser.parse_args()

    effective_model = args.model or (settings.api_model if args.llm_backend == "api" else None)
    api_key = settings.api_key.strip() if args.llm_backend == "api" else None

    retrieval_start = time.perf_counter()
    hits = _retrieve_results(args.question, args.top_k, args.retrieval_mode, args.rerank)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
    contexts = _build_contexts(hits)

    prompt = build_prompt(args.question, contexts, [context["chunk_id"] for context in contexts]) if contexts else ""

    generation_start = time.perf_counter()
    if contexts:
        client = create_llm_client(
            args.llm_backend,
            model=effective_model,
            api_key=api_key,
        )
        raw_payload = _generate_payload_with_retry(
            client=client,
            prompt=prompt,
            backend=args.llm_backend,
        )
    else:
        raw_payload = fallback_generation_payload()
    generation_latency_ms = (time.perf_counter() - generation_start) * 1000

    normalized = _normalize_answer_payload(raw_payload, contexts)
    pipeline = RAGPipeline(
        config=RAGConfig(
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            rerank=args.rerank,
            llm_backend=args.llm_backend,
            llm_model=effective_model,
            api_key=api_key,
        )
    )
    answer = RAGAnswer(
        answer=normalized["answer"],
        supporting_materials=normalized["supporting_materials"],
        explanation=normalized["explanation"],
        citations=normalized["citations"],
        retrieval_latency_ms=round(retrieval_latency_ms, 3),
        generation_latency_ms=round(generation_latency_ms, 3),
        retrieval_mode=args.retrieval_mode,
        llm_backend=args.llm_backend,
        verification=None,
    )
    verification = pipeline.verifier.verify(answer, contexts)
    if verification.corrected_output:
        corrected_payload = answer.model_dump()
        corrected_payload.update(verification.corrected_output)
        answer = RAGAnswer(**corrected_payload, verification=verification)
    else:
        answer = answer.model_copy(update={"verification": verification})

    print("=== RETRIEVED CONTEXTS ===")
    print(json.dumps(contexts, ensure_ascii=False, indent=2))
    print("\n=== PROMPT ===")
    print(prompt)
    print("\n=== RAW GENERATION PAYLOAD ===")
    print(json.dumps(raw_payload, ensure_ascii=False, indent=2))
    print("\n=== FINAL ANSWER JSON ===")
    print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("generation debug interrupted", file=sys.stderr)
        raise SystemExit(130)
