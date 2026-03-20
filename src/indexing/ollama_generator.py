"""Backward-compatible Ollama helper built on the unified LLM client."""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.generation.llm_client import (
    OllamaClient,
    fallback_generation_payload,
    normalize_generation_payload,
    parse_json_response,
)
from src.generation.prompt_templates import build_prompt

logger = get_logger(__name__)


def rag_ollama_answer(
    question: str,
    chunks_dict: dict[int, dict[str, Any]] | None,
    model: str | None = None,
    max_context_chars: int = 12000,
    temperature: float = 0.1,
    ollama_url: str = "http://localhost:11434",
) -> dict[str, Any]:
    """
    RAG-style generator using Ollama (free local LLM):
      - Takes a question and dict of retrieved chunks from get_chunks()
      - Builds context from chunks
      - Calls Ollama API for structured JSON response
      - Returns parsed answer with all required fields

    Args:
        question: The user's question
        chunks_dict: Output from get_chunks(), format:
            {
                1: {"chunk": "...", "paper": "...", "rank": 1},
                2: {"chunk": "...", "paper": "...", "rank": 2},
                ...
            }
        model: Ollama model name (llama2, mistral, llama3, etc.)
        max_context_chars: Maximum context length
        temperature: Model temperature (0.0-1.0)
        ollama_url: Base URL for Ollama API

    Returns:
        Dictionary with keys: answer, answer_value, answer_unit, ref_id,
        supporting_materials, explanation

    Requirements:
        - Ollama installed: https://ollama.ai/download
        - Model pulled: ollama pull llama2 (or mistral, llama3, etc.)
        - Ollama running: ollama serve (usually runs automatically)
    """

    if not chunks_dict:
        return fallback_generation_payload()

    ordered_items = sorted(chunks_dict.items(), key=lambda item: item[0])
    contexts: list[dict[str, Any]] = []
    candidate_ref_ids: list[str] = []
    for _, info in ordered_items:
        text = str(info.get("chunk", "")).strip()
        if not text:
            continue
        ref_id = str(info.get("paper", "")).strip()
        candidate_ref_ids.append(ref_id)
        contexts.append(
            {
                "ref_id": ref_id,
                "text": text[:max_context_chars],
                "page": info.get("page"),
                "headings": info.get("headings", []),
                "chunk_type": info.get("chunk_type", "text"),
            }
        )

    prompt = build_prompt(question, contexts, candidate_ref_ids)
    client = OllamaClient(
        model=model,
        base_url=ollama_url,
        temperature=temperature,
    )
    raw_text = client.generate(prompt)
    payload = parse_json_response(raw_text)
    if payload is None:
        logger.warning("ollama_response_parse_failed")
        return fallback_generation_payload()
    return normalize_generation_payload(payload)


# Example usage combining with the local retrieval function
if __name__ == "__main__":
    logger.info("ollama_generator_module", status="ready")
