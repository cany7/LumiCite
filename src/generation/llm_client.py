from __future__ import annotations

import json
import re
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

from src.config.settings import get_settings
from src.core.constants import FALLBACK_ANSWER
from src.core.errors import GenerationError, OllamaReadyError
from src.core.logging import get_logger, report_error, timed
from src.core.paths import find_project_root

logger = get_logger(__name__)
JSON_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$", re.MULTILINE)
INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')
GENERATION_SECTION_RE = re.compile(
    r"^(ANSWER|SUPPORTING_MATERIALS|EXPLANATION|CITED_CHUNK_IDS):[ \t]*(.*)$",
    re.MULTILINE,
)


def fallback_generation_payload() -> dict[str, Any]:
    return {
        "answer": FALLBACK_ANSWER,
        "supporting_materials": "is_blank",
        "explanation": "is_blank",
        "citations": [],
        "cited_chunk_ids": [],
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


def _loads_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _escape_invalid_json_escapes(text: str) -> str:
    return INVALID_JSON_ESCAPE_RE.sub(r"\\\\", text)


def parse_json_response(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    cleaned = strip_markdown_fences(text)
    candidates: list[str] = [cleaned]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        payload = _loads_json_object(candidate)
        if payload is not None:
            return payload

        repaired = _loads_json_object(_escape_invalid_json_escapes(candidate))
        if repaired is not None:
            return repaired

    return None


def _parse_chunk_id_lines(value: str) -> list[str]:
    chunk_ids: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line).strip()
        if not line:
            continue
        for candidate in [part.strip() for part in line.split(",")]:
            if candidate:
                chunk_ids.append(candidate)
    return chunk_ids


def parse_generation_response(text: str) -> dict[str, Any] | None:
    payload = parse_json_response(text)
    if payload is not None:
        return payload

    if not text:
        return None

    cleaned = strip_markdown_fences(text)
    matches = list(GENERATION_SECTION_RE.finditer(cleaned))
    if not matches:
        return None

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1).lower()
        inline_value = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        block_value = cleaned[start:end].strip()
        if inline_value and block_value:
            sections[label] = f"{inline_value}\n{block_value}".strip()
        else:
            sections[label] = inline_value or block_value

    if "answer" not in sections:
        return None

    return {
        "answer": sections.get("answer", "").strip(),
        "supporting_materials": sections.get("supporting_materials", "").strip(),
        "explanation": sections.get("explanation", "").strip(),
        "cited_chunk_ids": _parse_chunk_id_lines(sections.get("cited_chunk_ids", "")),
        "citations": [],
    }


def normalize_generation_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = fallback_generation_payload()
    payload.update(
        {
            "answer": str(data.get("answer", payload["answer"]) or payload["answer"]).strip(),
            "supporting_materials": str(
                data.get("supporting_materials", payload["supporting_materials"]) or payload["supporting_materials"]
            ).strip(),
            "explanation": str(data.get("explanation", payload["explanation"]) or payload["explanation"]).strip(),
            "citations": data.get("citations", payload["citations"]) or [],
            "cited_chunk_ids": data.get("cited_chunk_ids", payload["cited_chunk_ids"]) or [],
        }
    )

    if not isinstance(payload["citations"], list):
        payload["citations"] = []
    if not isinstance(payload["cited_chunk_ids"], list):
        payload["cited_chunk_ids"] = []

    return payload


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise NotImplementedError


def extract_chat_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = str(part.get("text", "")).strip()
            else:
                text = str(getattr(part, "text", "")).strip()
            if text:
                texts.append(text)
        return "\n".join(texts).strip()
    return ""


class APIClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float = 0.0,
        timeout: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model or settings.api_model
        self.api_key = (api_key if api_key is not None else settings.api_key).strip()
        self.base_url = (base_url if base_url is not None else settings.api_base_url).strip()
        self.reasoning_effort = reasoning_effort.strip().lower() if reasoning_effort else None
        self.temperature = temperature
        self.timeout = timeout or settings.api_timeout_seconds
        self._client: Any | None = None

    def _extra_body(self) -> dict[str, Any] | None:
        if not self.reasoning_effort:
            return None
        return {
            "reasoning": {
                "effort": self.reasoning_effort,
                "exclude": True,
            }
        }

    def _load_client(self) -> Any | None:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise GenerationError(
                error_type="api_key_missing",
                message="API key is required for generation",
                retryable=False,
                context={"model": self.model_name},
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenerationError(
                error_type="sdk_missing",
                message="OpenAI SDK is not installed",
                retryable=False,
                context={"model": self.model_name},
            ) from exc

        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise GenerationError(
                error_type="client_init_failed",
                message=str(exc),
                retryable=False,
                context={"model": self.model_name},
            ) from exc

        return self._client

    @timed("generate")
    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        client = self._load_client()
        effective_system_prompt = (
            system_prompt.strip()
            if system_prompt is not None and system_prompt.strip()
            else (
                "You answer RAG questions using only supplied evidence. "
                "Return only the requested structured plain-text fields."
            )
        )
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": effective_system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
        }
        extra_body = self._extra_body()
        if extra_body is not None:
            request_kwargs["extra_body"] = extra_body

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            raise GenerationError(
                error_type="generation_request_error",
                message=str(exc),
                retryable=True,
                context={"model": self.model_name},
            ) from exc

        text = extract_chat_response_text(response)
        if not text:
            raise GenerationError(
                error_type="generation_empty_output",
                message=f"Generation returned empty output for model {self.model_name}",
                retryable=True,
                context={"model": self.model_name},
            )
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
        self.model_name = model or settings.ollama_model or None
        self.temperature = temperature
        self.timeout = timeout

    def _resolve_model_name(self) -> str | None:
        if self.model_name:
            return self.model_name

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Ollama tags response must be a JSON object")
            models = payload.get("models", [])
        except requests.Timeout as exc:
            raise GenerationError(
                error_type="ollama_timeout",
                message=str(exc),
                retryable=True,
                context={"base_url": self.base_url},
            ) from exc
        except requests.RequestException as exc:
            raise GenerationError(
                error_type="ollama_request_error",
                message=str(exc),
                retryable=True,
                context={"base_url": self.base_url},
            ) from exc
        except ValueError as exc:
            raise GenerationError(
                error_type="ollama_parse_error",
                message=str(exc),
                retryable=True,
                context={"base_url": self.base_url},
            ) from exc

        if not models:
            raise GenerationError(
                error_type="ollama_request_error",
                message="No Ollama models are available",
                retryable=False,
                context={"base_url": self.base_url},
            )

        self.model_name = str(models[0].get("name", "")).strip() or None
        return self.model_name

    @timed("generate")
    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        model_name = self._resolve_model_name()
        if not model_name:
            raise GenerationError(
                error_type="ollama_request_error",
                message="Could not resolve an Ollama model",
                retryable=False,
                context={"base_url": self.base_url},
            )

        effective_prompt = prompt
        if system_prompt is not None and system_prompt.strip():
            effective_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": effective_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise GenerationError(
                error_type="ollama_timeout",
                message=str(exc),
                retryable=True,
                context={"model": model_name, "base_url": self.base_url},
            ) from exc
        except requests.RequestException as exc:
            raise GenerationError(
                error_type="ollama_request_error",
                message=str(exc),
                retryable=True,
                context={"model": model_name, "base_url": self.base_url},
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise GenerationError(
                error_type="ollama_parse_error",
                message=str(exc),
                retryable=True,
                context={"model": model_name, "base_url": self.base_url},
            ) from exc

        text = str(payload.get("response", "")).strip()
        if not text:
            raise GenerationError(
                error_type="ollama_empty_output",
                message=f"Ollama returned empty output for model {model_name}",
                retryable=True,
                context={"model": model_name, "base_url": self.base_url},
            )
        return text


def _ollama_status(base_url: str, model_name: str | None = None) -> tuple[bool, str | None]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return False, "ollama_unavailable"
    except ValueError:
        return False, "ollama_parse_error"

    if not isinstance(payload, dict):
        return False, "ollama_parse_error"

    if not model_name:
        return True, None

    normalized_model = model_name.strip()
    models = payload.get("models", [])
    if not isinstance(models, list):
        return False, "ollama_parse_error"

    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name == normalized_model:
            return True, None
    return False, "ollama_model_missing"


def _ollama_ready(base_url: str, model_name: str | None = None) -> bool:
    ready, _ = _ollama_status(base_url, model_name=model_name)
    return ready


def ensure_ollama_ready(
    *,
    base_url: str | None = None,
    compose_dir: str | Path | None = None,
    startup_timeout_seconds: int | None = None,
    poll_interval_seconds: float | None = None,
) -> str:
    settings = get_settings()
    effective_base_url = (base_url or settings.ollama_base_url).rstrip("/")
    effective_model = settings.ollama_model.strip() or None
    if _ollama_ready(effective_base_url, effective_model):
        logger.info("ollama_runtime_reused", base_url=effective_base_url, model=effective_model)
        return "reused"

    project_root = Path(compose_dir) if compose_dir is not None else find_project_root()
    command = ["docker", "compose", "up", "-d", "ollama"]
    result = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "docker compose failed"
        report_error(
            logger,
            "ollama_start_failed",
            "Ollama failed to start",
            error_type="ollama_start_failed",
            base_url=effective_base_url,
            detail=detail,
        )
        raise OllamaReadyError(
            error_type="ollama_start_failed",
            message=f"Failed to start ollama container: {detail}",
            retryable=False,
            context={"base_url": effective_base_url},
        )

    timeout_seconds = startup_timeout_seconds or settings.ollama_startup_timeout_seconds
    poll_seconds = poll_interval_seconds or settings.ollama_poll_interval_seconds
    deadline = time.monotonic() + timeout_seconds
    last_error_type = "ollama_ready_timeout"
    while time.monotonic() < deadline:
        ready, error_type = _ollama_status(effective_base_url, model_name=effective_model)
        if ready:
            logger.info("ollama_runtime_ready", base_url=effective_base_url, model=effective_model)
            return "started"
        if error_type:
            last_error_type = error_type
        time.sleep(poll_seconds)

    report_error(
        logger,
        last_error_type,
        "Ollama did not become ready in time" if last_error_type == "ollama_ready_timeout" else "Ollama is unavailable",
        error_type=last_error_type,
        base_url=effective_base_url,
        model=effective_model,
        timeout_seconds=timeout_seconds,
    )
    raise OllamaReadyError(
        error_type=last_error_type,
        message=(
            f"Ollama container did not become ready within {timeout_seconds} seconds"
            if last_error_type == "ollama_ready_timeout"
            else f"Ollama model {effective_model} is unavailable"
            if last_error_type == "ollama_model_missing"
            else "Ollama service is unavailable"
        ),
        retryable=False,
        context={"base_url": effective_base_url, "model": effective_model, "timeout_seconds": timeout_seconds},
    )


def create_llm_client(
    backend: str | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMClient:
    settings = get_settings()
    llm_backend = backend or settings.llm_backend
    if llm_backend == "ollama":
        return OllamaClient(model=model, base_url=base_url or settings.ollama_base_url)
    if llm_backend == "api":
        return APIClient(
            model=model or settings.api_model,
            api_key=api_key,
            base_url=base_url or settings.api_base_url,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError("llm backend must be one of: api, ollama")
