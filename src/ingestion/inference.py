from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from PIL import Image
import requests

from src.config.settings import get_settings
from src.core.errors import VisualInferenceError
from src.core.logging import get_logger
from src.core.paths import find_project_root

logger = get_logger(__name__)

VISUAL_SUMMARY_PROMPT = (
    "You are generating retrieval-oriented summaries for figures in academic papers. "
    "Convert the provided figure, together with its caption and any figure footnotes, into a compact English retrieval chunk for downstream RAG. "
    "Output English only. Return a single paragraph only. No bullet points, no numbering, no JSON, no markdown. "
    "Keep the output under 1000 characters, including spaces, and compress aggressively when needed. "
    "Base the summary on the visible content of the figure first. Use the caption and figure footnotes only to resolve labels, abbreviations, symbols, subfigure references, axes, units, legend items, experimental settings, and other conditions directly necessary to interpret this figure. "
    "Do not use external knowledge. "
    "Summarize at the highest retrieval-useful level first: identify the figure topic and the main thing the figure shows, compares, relates, organizes, or highlights. Preferred compression order: figure topic, dominant pattern or contrast, then only the most retrieval-useful supporting labels, entities, variables, metrics, values, modules, stages, axes, or constraints. "
    "When useful for retrieval, identify the figure form or analytical role, such as schematic, workflow, architecture diagram, result plot, ablation figure, heatmap, qualitative comparison, microscopy image, map, or other visual comparison. "
    "Prioritize retrieval-salient content: main entities, variables, components, stages, metrics, subfigure contrasts, explicit relationships, flows, rankings, trends, distributions, spatial patterns, interaction structure, notable values or ranges, units, and caption- or footnote-defined constraints. "
    "For comparative or multi-panel figures, state the overall comparison, dominant pattern, or division of roles before any panel-specific details, and include panel-level details only when they materially improve retrieval. Prefer cross-panel or overall patterns over isolated example values. Include specific values only when they are especially salient for retrieval or necessary to express a major contrast. "
    "For plots and charts, prioritize the main comparison, trend, trade-off, relative ordering, clustering, or dominant pattern over display details. Do not spend space on axis ranges, tick values, marker shapes, colors, hatching, or stylistic details unless they are necessary to express the main comparison, a meaningful scale difference, or retrieval value. "
    "You may state direct structural, comparative, spatial, or quantitative implications explicitly supported by the figure, caption, or footnotes, such as input-output relations, grouping, relative ordering, increasing or decreasing trends, clustering, overlap, separation, trade-offs, or panel-level contrasts. Prefer directly supported patterns over explanatory language about mechanism, benefit, effect, or motivation. "
    "Do not infer causes, motivations, methodology details beyond the figure, statistical significance, domain conclusions, or paper-level claims unless they are directly and explicitly supported by the figure, caption, or footnotes. "
    "Do not simply paraphrase the caption or footnotes. Do not describe the figure in reading order or narrate every box, arrow, marker, curve, region, object, or panel unless there is no higher-level pattern to summarize. "
    "Preserve technical terms, abbreviations, symbols, metric names, dataset names, model names, gene/protein/material names, anatomical terms, and units exactly as given. "
    "If any text, values, labels, legends, or visual details are unclear or unreadable, briefly note that instead of guessing. "
    "Be concise, factual, information-dense, and retrieval-optimized; prefer technical, noun-rich phrasing over narrative prose."
)


def _image_to_data_url(asset_path: Path) -> str:
    encoded = _image_to_base64(asset_path)
    return f"data:image/png;base64,{encoded}"


def _image_to_base64(asset_path: Path) -> str:
    try:
        with Image.open(asset_path) as image:
            working = image.convert("RGB")
            working.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            working.save(buffer, format="PNG")
    except Exception as exc:
        raise VisualInferenceError(
            error_type="image_missing",
            message=f"Image is missing or unreadable: {asset_path}",
            retryable=False,
            context={"path": str(asset_path), "detail": str(exc)},
        ) from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_response_text(response: object) -> str:
    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        response_text = response.get("response")
        if isinstance(response_text, str):
            return response_text.strip()
        raise VisualInferenceError(
            error_type="inference_parse_error",
            message="Visual inference response did not include message content",
            retryable=True,
        )

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise VisualInferenceError(
            error_type="inference_parse_error",
            message="Visual inference response did not include choices",
            retryable=True,
        )
    message = getattr(choices[0], "message", None)
    if message is None:
        raise VisualInferenceError(
            error_type="inference_parse_error",
            message="Visual inference response did not include a message",
            retryable=True,
        )
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
            else:
                text = str(getattr(item, "text", "")).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    raise VisualInferenceError(
        error_type="inference_parse_error",
        message="Visual inference response content had an unexpected shape",
        retryable=True,
    )


def _infer_with_api(
    *,
    client: object,
    settings: object,
    data_url: str,
    user_prompt: str,
    image_path: Path,
) -> str:
    try:
        response = client.chat.completions.create(
            model=settings.visual_api_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": VISUAL_SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
    except Exception as exc:
        raise VisualInferenceError(
            error_type="inference_request_error",
            message=str(exc),
            retryable=True,
            context={"path": str(image_path)},
        ) from exc

    text = _extract_response_text(response)
    if text:
        return text
    raise VisualInferenceError(
        error_type="inference_empty_output",
        message=f"Visual inference returned empty output for {image_path}",
        retryable=True,
        context={"path": str(image_path)},
    )


def _build_api_client(*, settings: object, image_path: Path) -> object:
    api_key = settings.visual_api_key.strip() or settings.api_key.strip()
    if not api_key:
        raise VisualInferenceError(
            error_type="inference_request_error",
            message="Visual inference API key is missing",
            retryable=False,
            context={"path": str(image_path)},
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise VisualInferenceError(
            error_type="inference_request_error",
            message="OpenAI SDK is not available for visual inference",
            retryable=False,
            context={"path": str(image_path)},
        ) from exc

    try:
        return OpenAI(
            api_key=api_key,
            base_url=(settings.visual_api_base_url or settings.api_base_url) or None,
            timeout=settings.visual_api_timeout_seconds,
        )
    except Exception as exc:
        raise VisualInferenceError(
            error_type="inference_request_error",
            message=str(exc),
            retryable=False,
            context={"path": str(image_path)},
        ) from exc


def _infer_with_ollama(
    *,
    settings: object,
    image_base64: str,
    user_prompt: str,
    image_path: Path,
) -> str:
    model_name = settings.ollama_model.strip()
    if not model_name:
        raise VisualInferenceError(
            error_type="ollama_request_error",
            message="Ollama model is not configured for visual inference",
            retryable=False,
            context={"path": str(image_path)},
        )

    try:
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat",
            json={
                "model": model_name,
                "stream": False,
                "messages": [
                    {"role": "system", "content": VISUAL_SUMMARY_PROMPT},
                    {"role": "user", "content": user_prompt, "images": [image_base64]},
                ],
                "options": {"temperature": 0.0},
            },
            timeout=settings.visual_api_timeout_seconds,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise VisualInferenceError(
            error_type="ollama_timeout",
            message=str(exc),
            retryable=True,
            context={"path": str(image_path), "base_url": settings.ollama_base_url},
        ) from exc
    except requests.RequestException as exc:
        raise VisualInferenceError(
            error_type="ollama_request_error",
            message=str(exc),
            retryable=True,
            context={"path": str(image_path), "base_url": settings.ollama_base_url},
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise VisualInferenceError(
            error_type="ollama_parse_error",
            message=str(exc),
            retryable=True,
            context={"path": str(image_path), "base_url": settings.ollama_base_url},
        ) from exc

    text = _extract_response_text(payload)
    if text:
        return text
    raise VisualInferenceError(
        error_type="ollama_empty_output",
        message=f"Ollama returned empty output for {image_path}",
        retryable=True,
        context={"path": str(image_path), "base_url": settings.ollama_base_url},
    )


def infer_figure_summary(
    *,
    asset_path: str,
    caption: str,
    footnotes: list[str],
    llm_backend: str | None = None,
) -> str:
    if not asset_path:
        raise VisualInferenceError(
            error_type="image_missing",
            message="Figure asset path is missing",
            retryable=False,
        )

    settings = get_settings()
    image_path = Path(asset_path)
    if not image_path.is_absolute():
        image_path = find_project_root() / image_path
    if not image_path.exists():
        raise VisualInferenceError(
            error_type="image_missing",
            message=f"Figure asset path does not exist: {image_path}",
            retryable=False,
            context={"path": str(image_path)},
        )

    backend = (llm_backend or getattr(settings, "llm_backend", "api") or "api").strip()
    if backend not in {"api", "ollama"}:
        raise VisualInferenceError(
            error_type="inference_request_error",
            message=f"Unsupported visual inference backend: {backend}",
            retryable=False,
            context={"path": str(image_path)},
        )
    image_base64 = _image_to_base64(image_path)
    data_url = f"data:image/png;base64,{image_base64}"
    footnote_text = "\n".join(footnotes)
    user_prompt = (
        "Figure/Table image is attached.\n\n"
        f"Caption:\n{caption or '(empty)'}\n\n"
        f"Footnotes:\n{footnote_text or '(empty)'}\n\n"
        "Return only the final summary text."
    )

    max_attempts = max(1, settings.request_retry_attempts)
    retry_delay = max(0.0, settings.request_retry_delay_seconds)
    api_client = _build_api_client(settings=settings, image_path=image_path) if backend == "api" else None
    last_error: VisualInferenceError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if backend == "ollama":
                text = _infer_with_ollama(
                    settings=settings,
                    image_base64=image_base64,
                    user_prompt=user_prompt,
                    image_path=image_path,
                )
            else:
                text = _infer_with_api(
                    client=api_client,
                    settings=settings,
                    data_url=data_url,
                    user_prompt=user_prompt,
                    image_path=image_path,
                )
            logger.info("visual_inference_succeeded", file_path=str(image_path), attempt=attempt, llm_backend=backend)
            return text
        except VisualInferenceError as exc:
            last_error = exc
            logger.warning(
                "visual_inference_failed",
                file_path=str(image_path),
                attempt=attempt,
                llm_backend=backend,
                error_type=exc.error_type,
                error=str(exc),
            )
            if not exc.retryable or attempt >= max_attempts:
                break
        except Exception as exc:
            last_error = VisualInferenceError(
                error_type="inference_request_error",
                message=str(exc),
                retryable=True,
                context={"path": str(image_path)},
            )
            logger.warning(
                "visual_inference_failed",
                file_path=str(image_path),
                attempt=attempt,
                llm_backend=backend,
                error=str(exc),
            )
            if attempt >= max_attempts:
                break
        if attempt < max_attempts:
            time.sleep(retry_delay)

    if last_error is None:
        last_error = VisualInferenceError(
            error_type="inference_request_error",
            message=f"Visual inference failed for {image_path}",
            retryable=False,
            context={"path": str(image_path)},
        )
    raise last_error
