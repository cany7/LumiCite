from __future__ import annotations

import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any

import requests

from src.config.settings import get_settings
from src.core.constants import FALLBACK_ANSWER
from src.core.logging import get_logger, timed

warnings.filterwarnings("ignore", category=UserWarning, module="google.cloud.aiplatform.initializer")

logger = get_logger(__name__)
JSON_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$", re.MULTILINE)


def fallback_generation_payload() -> dict[str, Any]:
    return {
        "answer": FALLBACK_ANSWER,
        "answer_value": "is_blank",
        "answer_unit": "is_blank",
        "ref_id": [],
        "supporting_materials": "is_blank",
        "explanation": "is_blank",
        "citations": [],
    }


def fallback_generation_json() -> str:
    return json.dumps(fallback_generation_payload(), ensure_ascii=False)


def strip_markdown_fences(text: str) -> str:
    if not text:
        return text
    match = JSON_FENCE_RE.search(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_response(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    cleaned = strip_markdown_fences(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None

    return payload if isinstance(payload, dict) else None


def normalize_generation_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = fallback_generation_payload()
    payload.update(
        {
            "answer": str(data.get("answer", payload["answer"]) or payload["answer"]).strip(),
            "answer_value": data.get("answer_value", payload["answer_value"]),
            "answer_unit": str(data.get("answer_unit", payload["answer_unit"]) or payload["answer_unit"]).strip(),
            "ref_id": data.get("ref_id", payload["ref_id"]),
            "supporting_materials": str(
                data.get("supporting_materials", payload["supporting_materials"]) or payload["supporting_materials"]
            ).strip(),
            "explanation": str(data.get("explanation", payload["explanation"]) or payload["explanation"]).strip(),
            "citations": data.get("citations", payload["citations"]) or [],
        }
    )

    if isinstance(payload["ref_id"], str):
        payload["ref_id"] = [payload["ref_id"]]
    if not isinstance(payload["ref_id"], list):
        payload["ref_id"] = []

    if not isinstance(payload["citations"], list):
        payload["citations"] = []

    return payload


def extract_vertex_response_text(response: Any) -> str:
    if not response:
        return ""
    if hasattr(response, "text") and response.text:
        return str(response.text).strip()

    candidates = getattr(response, "candidates", []) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) if content else []
        texts = [str(part.text).strip() for part in parts if getattr(part, "text", None)]
        if texts:
            return "\n".join(texts).strip()

    return ""


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        project: str | None = None,
        location: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model or settings.gemini_model
        self.project = project or settings.gcp_project
        self.location = location or settings.gcp_location
        self.credentials_path = credentials_path or settings.gcp_credentials_path
        self._client: Any | None = None

    def _load_client(self) -> Any | None:
        if self._client is not None:
            return self._client

        if self.credentials_path:
            if os.path.exists(self.credentials_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
            else:
                logger.warning("gemini_credentials_missing", path=self.credentials_path)
                return None

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            logger.warning("gemini_sdk_missing")
            return None

        try:
            vertexai.init(project=self.project, location=self.location)
            self._client = GenerativeModel(self.model_name)
        except Exception as exc:
            logger.warning("gemini_client_init_failed", error=str(exc), model=self.model_name)
            self._client = None

        return self._client

    @timed("generate")
    def generate(self, prompt: str) -> str:
        client = self._load_client()
        if client is None:
            return fallback_generation_json()

        try:
            response = client.generate_content(prompt, generation_config={"temperature": 0.0})
        except Exception as exc:
            logger.warning("gemini_generate_failed", error=str(exc), model=self.model_name)
            return fallback_generation_json()

        text = extract_vertex_response_text(response)
        if not text:
            logger.warning("gemini_generate_empty", model=self.model_name)
            return fallback_generation_json()
        return text


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model
        self.temperature = temperature
        self.timeout = timeout

    def _resolve_model_name(self) -> str | None:
        if self.model_name:
            return self.model_name

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
        except requests.RequestException as exc:
            logger.warning("ollama_model_discovery_failed", error=str(exc), base_url=self.base_url)
            return None

        if not models:
            logger.warning("ollama_no_models_available", base_url=self.base_url)
            return None

        self.model_name = str(models[0].get("name", "")).strip() or None
        return self.model_name

    @timed("generate")
    def generate(self, prompt: str) -> str:
        model_name = self._resolve_model_name()
        if not model_name:
            return fallback_generation_json()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("ollama_generate_failed", error=str(exc), model=model_name, base_url=self.base_url)
            return fallback_generation_json()

        text = str(response.json().get("response", "")).strip()
        if not text:
            logger.warning("ollama_generate_empty", model=model_name, base_url=self.base_url)
            return fallback_generation_json()
        return text


def create_llm_client(backend: str | None = None) -> LLMClient:
    settings = get_settings()
    llm_backend = backend or settings.llm_backend
    if llm_backend == "ollama":
        return OllamaClient(base_url=settings.ollama_base_url)
    if llm_backend == "gemini":
        return GeminiClient(
            model=settings.gemini_model,
            project=settings.gcp_project,
            location=settings.gcp_location,
            credentials_path=settings.gcp_credentials_path,
        )
    raise ValueError("llm backend must be one of: gemini, ollama")
