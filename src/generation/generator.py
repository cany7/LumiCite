"""Small Vertex AI Gemini wrapper that returns structured JSON answers."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
import warnings

# Suppress a specific warning from the Vertex AI SDK
warnings.filterwarnings("ignore", category=UserWarning, module="google.cloud.aiplatform.initializer")

FALLBACK_ANSWER = "Unable to answer with confidence based on the provided documents."


def _strip_md_fences(text: str) -> str:
    """Remove markdown fences if present."""
    if not text:
        return text
    fenced = re.compile(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$", re.MULTILINE)
    m = fenced.search(text.strip())
    if m:
        return m.group(1).strip()
    return text.strip()


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Best‑effort JSON parsing from model output."""
    if not text:
        return None
    text = _strip_md_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None


def _extract_response_text(response: Any) -> str:
    """Best-effort text extraction from Vertex AI Gemini responses."""
    if not response:
        return ""
    if hasattr(response, "text"):
        return response.text.strip()
    # Fallback for more complex response structures
    candidates = getattr(response, "candidates", []) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", []) if content else []
        texts = [part.text for part in parts if hasattr(part, "text")]
        if texts:
            return "\n".join(texts).strip()
    return ""


class LLMGenerator:
    """Vertex AI Gemini client that returns a dict with required fields."""

    def __init__(
        self,
        model: str,
        project: str,
        location: str,
        credentials_path: Optional[str] = None,
    ) -> None:
        self.model_name = model
        self.project = project
        self.location = location
        self.credentials_path = credentials_path
        self._client = None
        self._maybe_init()

    def _maybe_init(self) -> None:
        """Initialize Vertex AI client using the GOOGLE_APPLICATION_CREDENTIALS environment variable."""
        if self.credentials_path and os.path.exists(self.credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        else:
            warnings.warn(f"Credentials file not found at '{self.credentials_path}'. Authentication will likely fail.")
            return

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

        except ImportError:
            warnings.warn(
                "Vertex AI SDK not installed. Run `pip install google-cloud-aiplatform`. "
                "LLM generation will return fallback."
            )
            self._client = None
            return

        try:
            # The SDK will automatically find the credentials from the environment variable.
            vertexai.init(project=self.project, location=self.location)
            self._client = GenerativeModel(self.model_name)
            print("Vertex AI client initialized successfully.")
        except Exception as e:
            warnings.warn(f"Error initializing Vertex AI client: {e}")
            self._client = None

    def generate_json(self, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """Call the model; parse JSON; fallback on errors."""
        if self._client is None:
            warnings.warn("LLM client not initialized. Returning fallback answer.")
            return self._fallback()

        last_text: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.generate_content(prompt, generation_config={"temperature": 0.0})
                last_text = _extract_response_text(resp)
                if not last_text:
                    print(f"LLM call returned empty text (attempt {attempt}).")
                    continue
                
                data = _safe_json_loads(last_text)
                if isinstance(data, dict):
                    return self._normalize_output(data)
                else:
                    print(f"Failed to parse JSON from model output (attempt {attempt}).")

            except Exception as e:
                print(f"LLM call failed (attempt {attempt}): {e}")

        print("Falling back after repeated parse/generation errors.")
        if last_text:
            print("Last raw model text (truncated):", last_text[:300])
        return self._fallback()

    def _normalize_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required fields exist and types are consistent."""
        def get(k: str, default: Any) -> Any:
            v = data.get(k, default)
            return v if v is not None else default

        norm = {
            "answer": get("answer", ""),
            "answer_value": get("answer_value", ""),
            "answer_unit": get("answer_unit", ""),
            "ref_id": get("ref_id", []),
            "supporting_materials": get("supporting_materials", ""),
            "explanation": get("explanation", ""),
        }
        if isinstance(norm["ref_id"], str):
            norm["ref_id"] = [norm["ref_id"]]
        if not isinstance(norm["ref_id"], list):
            norm["ref_id"] = []
        for k in ("answer", "answer_unit", "supporting_materials", "explanation"):
            if isinstance(norm[k], str):
                norm[k] = norm[k].strip()
        return norm

    def _fallback(self) -> Dict[str, Any]:
        return {
            "answer": FALLBACK_ANSWER,
            "answer_value": "is_blank",
            "answer_unit": "is_blank",
            "ref_id": [],
            "supporting_materials": "is_blank",
            "explanation": "is_blank",
        }
