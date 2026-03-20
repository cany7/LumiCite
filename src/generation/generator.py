"""Backward-compatible Gemini JSON wrapper built on the unified LLM client."""
from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.generation.llm_client import (
    GeminiClient,
    fallback_generation_payload,
    normalize_generation_payload,
    parse_json_response,
)

logger = get_logger(__name__)


class LLMGenerator:
    """Compatibility wrapper that keeps the old `generate_json()` interface."""

    def __init__(
        self,
        model: str,
        project: str,
        location: str,
        credentials_path: str | None = None,
    ) -> None:
        self.client = GeminiClient(
            model=model,
            project=project,
            location=location,
            credentials_path=credentials_path,
        )

    def generate_json(self, prompt: str, max_retries: int = 2) -> dict[str, Any]:
        last_text = ""
        for attempt in range(1, max_retries + 1):
            last_text = self.client.generate(prompt)
            payload = parse_json_response(last_text)
            if payload is not None:
                return normalize_generation_payload(payload)
            logger.warning("gemini_response_parse_failed", attempt=attempt)

        logger.warning("gemini_response_fallback")
        if last_text:
            logger.warning("gemini_response_last_text", preview=last_text[:300])
        return fallback_generation_payload()
