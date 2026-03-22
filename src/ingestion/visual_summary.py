from __future__ import annotations

from src.ingestion.inference import infer_figure_summary


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _normalize_footnotes(footnotes: list[str]) -> list[str]:
    normalized: list[str] = []
    for footnote in footnotes:
        text = _normalize_text(footnote)
        if text:
            normalized.append(text)
    return normalized


def generate_figure_summary(
    *,
    asset_path: str,
    caption: str,
    footnotes: list[str],
    llm_backend: str | None = None,
) -> str:
    return infer_figure_summary(
        asset_path=asset_path,
        caption=_normalize_text(caption),
        footnotes=_normalize_footnotes(footnotes),
        llm_backend=llm_backend,
    )


def build_figure_text(*, summary: str, caption: str, footnotes: list[str]) -> str:
    normalized_summary = _normalize_text(summary)
    normalized_caption = _normalize_text(caption)
    normalized_footnotes = _normalize_footnotes(footnotes)

    parts: list[str] = [normalized_summary] if normalized_summary else []
    if normalized_caption:
        parts.append(f"Caption: {normalized_caption}")
    if normalized_footnotes:
        parts.append(f"Footnotes: {' '.join(normalized_footnotes)}")
    return "\n".join(parts).strip()


def linearize_table_text(*, body_text: str, caption: str, footnotes: list[str]) -> str:
    normalized_body = _normalize_text(body_text)
    normalized_caption = _normalize_text(caption)
    normalized_footnotes = _normalize_footnotes(footnotes)

    parts: list[str] = []
    if normalized_caption:
        parts.append(f"Table caption: {normalized_caption}")
    if normalized_body:
        parts.append(f"Table body: {normalized_body}")
    if normalized_footnotes:
        parts.append(f"Footnotes: {' '.join(normalized_footnotes)}")
    return "\n".join(parts).strip()
