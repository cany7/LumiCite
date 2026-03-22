from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.ingestion.mineru_mapper as mapper_module
from src.core.errors import VisualInferenceError
from src.core.schemas import FigureChunk, TableChunk, TextChunk
from src.ingestion.chunker import TextBlock, split_text_blocks
from src.ingestion.mineru_mapper import map_mineru_output


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_png(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), color=(10, 20, 30)).save(path)
    return path


def test_split_text_blocks_merge_same_headings_across_pages_and_split_on_heading_change() -> None:
    chunks = split_text_blocks(
        "paper1",
        [
            TextBlock(text="Alpha page one.", page_number=1, headings=["1 Intro"]),
            TextBlock(text="Beta page two.", page_number=2, headings=["1 Intro"]),
            TextBlock(text="Gamma result.", page_number=2, headings=["2 Results"]),
        ],
        chunk_size=1000,
        chunk_overlap=100,
    )

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].headings == ["1 Intro"]
    assert chunks[0].text == "Alpha page one. Beta page two."
    assert chunks[1].headings == ["2 Results"]


def test_map_mineru_output_builds_current_chunk_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    output_dir = root / "data" / "intermediate" / "mineru" / "paper1"
    images_dir = output_dir / "images"
    _write_png(images_dir / "figure.png")
    _write_png(images_dir / "table.png")
    content_list_path = _write_json(
        output_dir / "paper1_content_list.json",
        [
            {"type": "title", "text": "3 Results", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Alpha paragraph.", "page_idx": 0, "bbox": [0, 1, 2, 3]},
            {"type": "list", "text": "Beta bullet.", "page_idx": 1, "poly": [1, 2, 3]},
            {
                "type": "image",
                "img_path": "figure.png",
                "image_caption": "Figure 1. Accuracy trend",
                "image_footnote": ["normalized"],
                "page_idx": 1,
                "bbox": [0, 1, 2, 3],
            },
            {
                "type": "table",
                "img_path": "table.png",
                "table_caption": "Table 1. Scores",
                "table_footnote": ["macro average"],
                "table_body": "Model A 0.9 Model B 0.8",
                "page_idx": 2,
            },
        ],
    )
    middle_json_path = _write_json(output_dir / "paper1_middle.json", {})
    captured_summary_calls: list[dict[str, object]] = []

    def fake_generate_figure_summary(
        *,
        asset_path: str,
        caption: str,
        footnotes: list[str],
        llm_backend: str | None = None,
    ) -> str:
        captured_summary_calls.append(
            {"asset_path": asset_path, "caption": caption, "footnotes": footnotes, "llm_backend": llm_backend}
        )
        return "Figure summary"

    monkeypatch.setattr(mapper_module, "get_settings", lambda: SimpleNamespace(chunk_size=1000, chunk_overlap=100))
    monkeypatch.setattr(mapper_module, "generate_figure_summary", fake_generate_figure_summary)

    chunks = map_mineru_output(
        doc_id="paper1",
        content_list_path=content_list_path,
        middle_json_path=middle_json_path,
        raw_images_dir=images_dir,
        output_dir=output_dir,
        root=root,
    )

    text_chunk = next(chunk for chunk in chunks if isinstance(chunk, TextChunk))
    figure_chunk = next(chunk for chunk in chunks if isinstance(chunk, FigureChunk))
    table_chunk = next(chunk for chunk in chunks if isinstance(chunk, TableChunk))

    assert text_chunk.headings == ["3 Results"]
    assert text_chunk.page_number == 1
    assert text_chunk.text == "Alpha paragraph. Beta bullet."
    assert figure_chunk.text == "Figure summary\nCaption: Figure 1. Accuracy trend\nFootnotes: normalized"
    assert figure_chunk.caption == "Figure 1. Accuracy trend"
    assert figure_chunk.asset_path.startswith("data/assets/paper1/paper1_fig_")
    assert table_chunk.text == (
        "Table caption: Table 1. Scores\n"
        "Table body: Model A 0.9 Model B 0.8\n"
        "Footnotes: macro average"
    )
    assert table_chunk.asset_path == ""
    assert set(captured_summary_calls[0]) == {"asset_path", "caption", "footnotes", "llm_backend"}
    assert captured_summary_calls[0]["caption"] == "Figure 1. Accuracy trend"
    assert captured_summary_calls[0]["footnotes"] == ["normalized"]
    assert captured_summary_calls[0]["llm_backend"] is None
    assert "bbox" not in figure_chunk.model_dump()
    assert "poly" not in text_chunk.model_dump()


def test_map_mineru_output_skips_figure_when_visual_summary_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    output_dir = root / "data" / "intermediate" / "mineru" / "paper1"
    images_dir = output_dir / "images"
    _write_png(images_dir / "figure.png")
    content_list_path = _write_json(
        output_dir / "paper1_content_list.json",
        [
            {"type": "title", "text": "3 Results", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Alpha paragraph.", "page_idx": 0},
            {
                "type": "image",
                "img_path": "figure.png",
                "image_caption": "Figure 1. Accuracy trend",
                "image_footnote": ["normalized"],
                "page_idx": 1,
            },
        ],
    )
    middle_json_path = _write_json(output_dir / "paper1_middle.json", {})
    reported: list[tuple[str, str, dict[str, object]]] = []

    def fake_report_error(logger, event: str, cli_message: str, **fields):  # noqa: ANN001
        reported.append((event, cli_message, fields))

    def fail_generate_figure_summary(
        *,
        asset_path: str,
        caption: str,
        footnotes: list[str],
        llm_backend: str | None = None,
    ) -> str:
        raise VisualInferenceError(
            error_type="inference_empty_output",
            message="empty summary",
            retryable=True,
            context={"path": asset_path},
        )

    monkeypatch.setattr(mapper_module, "get_settings", lambda: SimpleNamespace(chunk_size=1000, chunk_overlap=100))
    monkeypatch.setattr(mapper_module, "generate_figure_summary", fail_generate_figure_summary)
    monkeypatch.setattr(mapper_module, "report_error", fake_report_error)

    chunks = map_mineru_output(
        doc_id="paper1",
        content_list_path=content_list_path,
        middle_json_path=middle_json_path,
        raw_images_dir=images_dir,
        output_dir=output_dir,
        root=root,
    )

    assert [type(chunk) for chunk in chunks] == [TextChunk]
    assert reported[0][0] == "figure_chunk_skipped"
    assert reported[0][2]["error_type"] == "inference_empty_output"
