from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.core.paths import find_project_root
from src.core.schemas import ChunkType, FigureChunk, TableChunk

logger = get_logger(__name__)

CHART_TYPES = [
    "line chart",
    "bar chart",
    "scatter plot",
    "scatter chart",
    "plot",
    "diagram",
    "heatmap",
    "pie chart",
    "histogram",
]
TREND_HINTS = ("increase", "decrease", "higher", "lower", "trend", "plateau", "compare", "correlation")


def _infer_visual_name(entry: dict[str, Any]) -> str:
    headings = entry.get("headings", []) or []
    if headings:
        return str(headings[0])
    chunk_id = str(entry.get("chunk_id", ""))
    if "_figure" in chunk_id or "_table" in chunk_id:
        return chunk_id.rsplit("_", 1)[0]
    return chunk_id


def _infer_chunk_type(entry: dict[str, Any]) -> str:
    text = f"{_infer_visual_name(entry)} {entry.get('text', '')}".lower()
    if "_table" in text or "table " in text or text.startswith("table"):
        return "table"
    return "figure"


def _normalize_chunk_id(doc_id: str, chunk_id: str, chunk_type: str) -> str:
    suffix = chunk_id.rsplit("_", 1)[-1]
    label = "tab" if chunk_type == "table" else "fig"
    if re.fullmatch(r"[0-9a-f]{8}", suffix):
        return f"{doc_id}_{label}_{suffix}"
    return f"{doc_id}_{label}_00000000"


def _infer_chart_type(text: str) -> str:
    lowered = text.lower()
    for chart_type in CHART_TYPES:
        if chart_type in lowered:
            return chart_type
    return ""


def _infer_axis_labels(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    x_match = re.search(r"x-axis[^.]*represents\s+([^.;]+)", text, re.IGNORECASE)
    y_match = re.search(r"y-axis[^.]*represents\s+([^.;]+)", text, re.IGNORECASE)
    if x_match:
        labels["x"] = x_match.group(1).strip()
    if y_match:
        labels["y"] = y_match.group(1).strip()
    return labels


def _infer_trends(text: str) -> list[str]:
    trends: list[str] = []
    for sentence in re.split(r"(?<=[.?!])\s+", text):
        normalized = sentence.strip()
        lowered = normalized.lower()
        if normalized and any(hint in lowered for hint in TREND_HINTS):
            trends.append(normalized)
        if len(trends) == 3:
            break
    return trends


def _infer_key_values(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z][A-Za-z0-9 #()/:-]{1,40})\s*=\s*([^.;,\n]{1,80})", text):
        pairs[key.strip()] = value.strip()
        if len(pairs) == 5:
            break
    return pairs


def _build_table_chunk(doc_id: str, entry: dict[str, Any]) -> TableChunk:
    visual_name = _infer_visual_name(entry)
    text = str(entry.get("text", "")).strip()
    return TableChunk(
        chunk_id=_normalize_chunk_id(doc_id, str(entry.get("chunk_id", "")), "table"),
        doc_id=doc_id,
        text=text,
        chunk_type=ChunkType.TABLE,
        page_number=entry.get("page_number"),
        headings=list(entry.get("headings", []) or [visual_name]),
        source_file=str(entry.get("source_file", f"{doc_id}.pdf")),
        column_headers=[],
        row_headers=[],
        key_values=_infer_key_values(text),
        caption=visual_name,
    )


def _build_figure_chunk(doc_id: str, entry: dict[str, Any]) -> FigureChunk:
    visual_name = _infer_visual_name(entry)
    text = str(entry.get("text", "")).strip()
    return FigureChunk(
        chunk_id=_normalize_chunk_id(doc_id, str(entry.get("chunk_id", "")), "figure"),
        doc_id=doc_id,
        text=text,
        chunk_type=ChunkType.FIGURE,
        page_number=entry.get("page_number"),
        headings=list(entry.get("headings", []) or [visual_name]),
        source_file=str(entry.get("source_file", f"{doc_id}.pdf")),
        chart_type=_infer_chart_type(text),
        axis_labels=_infer_axis_labels(text),
        trends=_infer_trends(text),
        caption=visual_name,
    )


def upgrade_alt_text_data(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    normalized_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for doc_id, entries in data.items():
        upgraded: list[dict[str, Any]] = []
        for entry in entries:
            chunk_type = _infer_chunk_type(entry)
            if chunk_type == "table":
                upgraded.append(_build_table_chunk(doc_id, entry).model_dump())
            else:
                upgraded.append(_build_figure_chunk(doc_id, entry).model_dump())

        upgraded.sort(
            key=lambda item: (
                item.get("chunk_type") != "table",
                item.get("page_number") is None,
                item.get("page_number") or 0,
                item.get("caption", ""),
            )
        )
        normalized_groups.append((doc_id, upgraded))

    normalized_groups.sort(key=lambda item: (not any(entry["chunk_type"] == "table" for entry in item[1]), item[0]))
    return {doc_id: entries for doc_id, entries in normalized_groups}


def upgrade_alt_text_file(path: Path | None = None) -> Path:
    root = find_project_root()
    target = path or (root / "data" / "JSON" / "alt_text.json")
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    upgraded = upgrade_alt_text_data(data)
    target.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("alt_text_upgraded", path=str(target), documents=len(upgraded))
    return target


def batch_generate_alt_text() -> Path:
    return upgrade_alt_text_file()


if __name__ == "__main__":
    batch_generate_alt_text()
