from __future__ import annotations

import base64
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import src.ingestion.inference as inference_module
import src.ingestion.visual_summary as visual_summary_module
from src.ingestion.inference import VISUAL_SUMMARY_PROMPT, _image_to_data_url, infer_figure_summary
from src.ingestion.visual_assets import copy_asset_to_canonical, resolve_raw_asset_path
from src.ingestion.visual_summary import build_figure_text, generate_figure_summary, linearize_table_text


def _write_image(path: Path, size: tuple[int, int] = (1600, 900)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(12, 34, 56)).save(path)
    return path


def test_visual_asset_resolution_and_copy_follow_canonical_layout(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "data").mkdir()
    output_dir = root / "data" / "intermediate" / "mineru" / "paper1"
    raw_image = _write_image(output_dir / "images" / "figure.jpg", size=(300, 200))

    resolved = resolve_raw_asset_path("figure.jpg", output_dir=output_dir, raw_images_dir=output_dir / "images")
    copied = copy_asset_to_canonical("paper1", "paper1_fig_deadbeef", resolved, root=root)

    assert resolved == raw_image
    assert copied == "data/assets/paper1/paper1_fig_deadbeef.jpg"
    assert (root / copied).exists()


def test_image_to_data_url_resizes_longest_edge_to_1024(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "wide.png", size=(3000, 1000))

    data_url = _image_to_data_url(image_path)

    assert data_url.startswith("data:image/png;base64,")
    encoded = data_url.split(",", 1)[1]
    payload = base64.b64decode(encoded)
    with Image.open(io.BytesIO(payload)) as resized:
        assert max(resized.size) == 1024


def test_infer_figure_summary_retries_on_empty_output_and_builds_multimodal_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _write_image(tmp_path / "figure.png", size=(1200, 800))
    settings = SimpleNamespace(
        visual_api_key="visual-key",
        api_key="",
        visual_api_base_url="",
        api_base_url="",
        visual_api_timeout_seconds=15,
        visual_api_model="vision-model",
        request_retry_attempts=2,
        request_retry_delay_seconds=0.0,
    )
    calls: list[dict] = []

    def response_with_text(text: str) -> object:
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        )

    class FakeClient:
        def __init__(self) -> None:
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
            self._responses = [response_with_text(""), response_with_text("A figure compares emissions trends.")]

        def create(self, **kwargs):  # noqa: ANN003
            calls.append(kwargs)
            return self._responses.pop(0)

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = lambda **kwargs: FakeClient()  # noqa: E731

    monkeypatch.setattr(inference_module, "get_settings", lambda: settings)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    summary = infer_figure_summary(
        asset_path=str(image_path),
        caption="Figure 1. Emissions trend",
        footnotes=["values are normalized"],
    )

    assert summary == "A figure compares emissions trends."
    assert len(calls) == 2
    assert calls[0]["model"] == "vision-model"
    assert calls[0]["messages"][0]["content"] == VISUAL_SUMMARY_PROMPT
    user_content = calls[0]["messages"][1]["content"]
    assert "Caption:\nFigure 1. Emissions trend" in user_content[0]["text"]
    assert "Footnotes:\nvalues are normalized" in user_content[0]["text"]
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_generate_figure_summary_and_text_builders_normalize_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_infer_figure_summary(
        *,
        asset_path: str,
        caption: str,
        footnotes: list[str],
        llm_backend: str | None = None,
    ) -> str:
        captured["asset_path"] = asset_path
        captured["caption"] = caption
        captured["footnotes"] = footnotes
        captured["llm_backend"] = llm_backend
        return " summary body "

    monkeypatch.setattr(visual_summary_module, "infer_figure_summary", fake_infer_figure_summary)

    summary = generate_figure_summary(
        asset_path="data/assets/paper1/figure.png",
        caption=" Figure 1. Trend ",
        footnotes=[" note a ", "", " note b "],
    )
    figure_text = build_figure_text(summary=summary, caption=" Figure 1. Trend ", footnotes=[" note a ", "note b"])
    table_text = linearize_table_text(
        body_text=" Model A 0.9 ",
        caption=" Table 1. Scores ",
        footnotes=[" macro avg "],
    )

    assert captured == {
        "asset_path": "data/assets/paper1/figure.png",
        "caption": "Figure 1. Trend",
        "footnotes": ["note a", "note b"],
        "llm_backend": None,
    }
    assert figure_text == "summary body\nCaption: Figure 1. Trend\nFootnotes: note a note b"
    assert table_text == "Table caption: Table 1. Scores\nTable body: Model A 0.9\nFootnotes: macro avg"


def test_infer_figure_summary_uses_ollama_chat_for_multimodal_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _write_image(tmp_path / "figure.png", size=(1200, 800))
    settings = SimpleNamespace(
        llm_backend="ollama",
        request_retry_attempts=1,
        request_retry_delay_seconds=0.0,
        visual_api_timeout_seconds=21,
        ollama_base_url="http://ollama.local:11434",
        ollama_model="qwen3.5:4b",
    )
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {"role": "assistant", "content": "A figure shows a rising trend."}}

    def fake_post(url: str, **kwargs):  # noqa: ANN003
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(inference_module, "get_settings", lambda: settings)
    monkeypatch.setattr(inference_module.requests, "post", fake_post)

    summary = infer_figure_summary(
        asset_path=str(image_path),
        caption="Figure 2. Rising trend",
        footnotes=["normalized values"],
        llm_backend="ollama",
    )

    assert summary == "A figure shows a rising trend."
    assert calls[0]["url"] == "http://ollama.local:11434/api/chat"
    assert calls[0]["timeout"] == 21
    assert calls[0]["json"]["model"] == "qwen3.5:4b"
    assert calls[0]["json"]["messages"][0]["content"] == VISUAL_SUMMARY_PROMPT
    assert "Caption:\nFigure 2. Rising trend" in calls[0]["json"]["messages"][1]["content"]
    assert calls[0]["json"]["messages"][1]["images"][0]
    assert not calls[0]["json"]["messages"][1]["images"][0].startswith("data:image/")
