from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.core.errors import VisualInferenceError
from src.core.logging import get_logger, report_error
from src.core.paths import find_project_root
from src.core.schemas import ChunkModel, ChunkType, FigureChunk, TableChunk
from src.ingestion.chunker import TextBlock, build_chunk_id, split_text_blocks
from src.ingestion.visual_assets import copy_asset_to_canonical, resolve_raw_asset_path
from src.ingestion.visual_summary import build_figure_text, generate_figure_summary, linearize_table_text

logger = get_logger(__name__)

VISUAL_IGNORE_KEYS = {
    "bbox",
    "poly",
    "img_path",
    "image_path",
    "page_idx",
    "page_number",
    "text_level",
    "type",
    "image_caption",
    "image_footnote",
    "table_caption",
    "table_footnote",
    "footnotes",
    "footnote",
    "caption",
}


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_content_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        dict_items = [item for item in payload if isinstance(item, dict)]
        if dict_items and any("type" in item for item in dict_items):
            return dict_items
        for item in payload:
            found = _extract_content_items(item)
            if found:
                return found

    if isinstance(payload, dict):
        for key in ("content_list", "items", "data", "content", "contents", "result"):
            if key in payload:
                found = _extract_content_items(payload[key])
                if found:
                    return found
        for value in payload.values():
            found = _extract_content_items(value)
            if found:
                return found

    return []


def _collect_middle_records(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            records.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return records


def _build_middle_index(payload: Any) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for record in _collect_middle_records(payload):
        for key in ("id", "block_id", "anno_id", "img_path", "image_path", "path"):
            value = str(record.get(key, "")).strip()
            if not value:
                continue
            index.setdefault(value, []).append(record)
            index.setdefault(Path(value).name, []).append(record)
    return index


def _middle_matches(item: dict[str, Any], index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    keys: list[str] = []
    for key in ("id", "block_id", "anno_id", "img_path", "image_path", "path"):
        value = str(item.get(key, "")).strip()
        if not value:
            continue
        keys.append(value)
        keys.append(Path(value).name)

    matches: list[dict[str, Any]] = []
    seen: set[int] = set()
    for key in keys:
        for record in index.get(key, []):
            marker = id(record)
            if marker in seen:
                continue
            seen.add(marker)
            matches.append(record)
    return matches


def _text_from_value(value: Any, *, ignore_keys: set[str] | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _normalize_text(str(value))
    if isinstance(value, list):
        parts = [_text_from_value(item, ignore_keys=ignore_keys) for item in value]
        return _normalize_text(" ".join(part for part in parts if part))
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if ignore_keys and key in ignore_keys:
                continue
            text = _text_from_value(item, ignore_keys=ignore_keys)
            if text:
                parts.append(text)
        return _normalize_text(" ".join(parts))
    return ""


def _list_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items: list[str] = []
        for entry in value:
            items.extend(_list_from_value(entry))
        return items
    text = _text_from_value(value)
    if not text:
        return []
    if "\n" in text:
        return [_normalize_text(part) for part in text.splitlines() if _normalize_text(part)]
    return [text]


def _first_text(sources: list[dict[str, Any]], keys: list[str], *, ignore_keys: set[str] | None = None) -> str:
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            text = _text_from_value(source[key], ignore_keys=ignore_keys)
            if text:
                return text
    return ""


def _first_list(sources: list[dict[str, Any]], keys: list[str]) -> list[str]:
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            values = _list_from_value(source[key])
            if values:
                return values
    return []


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_number(item: dict[str, Any], supplemental: list[dict[str, Any]]) -> int | None:
    for source in [item, *supplemental]:
        page_idx = _coerce_int(source.get("page_idx"))
        if page_idx is not None:
            return page_idx + 1
        page_number = _coerce_int(source.get("page_number"))
        if page_number is not None:
            return page_number
        page = _coerce_int(source.get("page"))
        if page is not None:
            return page
    return None


def _update_heading_stack(headings: list[str], title_text: str, text_level: Any) -> list[str]:
    title = _normalize_text(title_text)
    if not title:
        return headings

    level = _coerce_int(text_level)
    if level is None or level <= 0:
        return [*headings, title]

    next_headings = headings[: max(0, level - 1)]
    next_headings.append(title)
    return next_headings


def _fallback_body_text(sources: list[dict[str, Any]]) -> str:
    for source in sources:
        text = _text_from_value(source, ignore_keys=VISUAL_IGNORE_KEYS)
        if text:
            return text
    return ""


def map_mineru_output(
    *,
    doc_id: str,
    content_list_path: str | Path,
    middle_json_path: str | Path | None = None,
    raw_images_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    root: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    llm_backend: str | None = None,
) -> list[ChunkModel]:
    settings = get_settings()
    project_root = root or find_project_root()
    content_path = Path(content_list_path)
    output_path = Path(output_dir) if output_dir is not None else content_path.parent
    raw_images_path = Path(raw_images_dir) if raw_images_dir is not None else output_path / "images"

    content_items = _extract_content_items(_load_json(content_path))
    if not content_items:
        raise ValueError(f"No content_list items found in {content_path}")

    middle_index: dict[str, list[dict[str, Any]]] = {}
    if middle_json_path:
        middle_path = Path(middle_json_path)
        if middle_path.exists():
            middle_index = _build_middle_index(_load_json(middle_path))

    headings: list[str] = []
    text_blocks: list[TextBlock] = []
    visual_chunks: list[ChunkModel] = []

    for item in content_items:
        block_type = _normalize_text(str(item.get("type", "")).lower())
        supplemental = _middle_matches(item, middle_index)
        page_number = _page_number(item, supplemental)

        if block_type == "title":
            title_text = _first_text([item, *supplemental], ["text", "content", "title", "value"])
            headings = _update_heading_stack(headings, title_text, item.get("text_level"))
            continue

        current_headings = list(headings)
        if block_type in {"text", "list"}:
            text = _first_text([item, *supplemental], ["text", "content", "body", "markdown"])
            if not text:
                text = _fallback_body_text([item, *supplemental])
            if text:
                text_blocks.append(TextBlock(text=text, page_number=page_number, headings=current_headings))
            continue

        if block_type not in {"image", "table"}:
            continue

        is_table = block_type == "table"
        caption = _first_text(
            [item, *supplemental],
            ["table_caption", "caption", "title"] if is_table else ["image_caption", "figure_caption", "caption", "title"],
        )
        footnotes = _first_list(
            [item, *supplemental],
            ["table_footnote", "footnotes", "footnote"] if is_table else ["image_footnote", "footnotes", "footnote"],
        )
        raw_asset_ref = _first_text([item, *supplemental], ["img_path", "image_path", "path"])
        raw_asset_path = resolve_raw_asset_path(
            raw_asset_ref,
            output_dir=output_path,
            raw_images_dir=raw_images_path,
        )

        chunk_type = ChunkType.TABLE if is_table else ChunkType.FIGURE
        body = ""
        if is_table:
            body = _first_text(
                [item, *supplemental],
                ["table_body", "table_text", "body", "html", "markdown", "text"],
                ignore_keys=VISUAL_IGNORE_KEYS,
            )
            if not body:
                body = _fallback_body_text([*supplemental, item])
        chunk_id = build_chunk_id(
            doc_id,
            chunk_type,
            f"{page_number}|{caption}|{'|'.join(footnotes)}|{raw_asset_ref}|{body}",
        )
        if is_table:
            text = linearize_table_text(body_text=body, caption=caption, footnotes=footnotes)
            visual_chunks.append(
                TableChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    page_number=page_number,
                    headings=current_headings,
                    caption=caption,
                    footnotes=footnotes,
                    asset_path="",
                )
            )
        else:
            try:
                asset_path = copy_asset_to_canonical(doc_id, chunk_id, raw_asset_path, root=project_root)
                summary = generate_figure_summary(
                    asset_path=asset_path,
                    caption=caption,
                    footnotes=footnotes,
                    llm_backend=llm_backend,
                )
                text = build_figure_text(summary=summary, caption=caption, footnotes=footnotes)
            except (FileNotFoundError, VisualInferenceError) as exc:
                error_type = "image_missing" if isinstance(exc, FileNotFoundError) else exc.error_type
                report_error(
                    logger,
                    "figure_chunk_skipped",
                    f"Skipped figure for {doc_id} ({error_type})",
                    doc_id=doc_id,
                    path=str(raw_asset_path),
                    error_type=error_type,
                    detail=str(exc),
                )
                continue
            visual_chunks.append(
                FigureChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    page_number=page_number,
                    headings=current_headings,
                    caption=caption,
                    footnotes=footnotes,
                    asset_path=asset_path,
                )
            )

    text_chunks = split_text_blocks(
        doc_id,
        text_blocks,
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )
    chunks = [*text_chunks, *visual_chunks]
    chunks.sort(key=lambda chunk: (chunk.page_number or 0, chunk.chunk_type.value, chunk.chunk_id))
    logger.info(
        "mineru_chunks_mapped",
        doc_id=doc_id,
        num_text_chunks=len(text_chunks),
        num_visual_chunks=len(visual_chunks),
        total_chunks=len(chunks),
    )
    return chunks
