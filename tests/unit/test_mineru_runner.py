from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.ingestion.mineru_runner as runner_module
from src.core.errors import MinerUProcessError
from src.ingestion.mineru_runner import run_local_mineru


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        mineru_model_source="huggingface",
        mineru_command="mineru-bin",
        mineru_timeout_seconds=12,
    )


def _write_outputs(output_dir: Path, doc_id: str) -> None:
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / f"{doc_id}_content_list.json").write_text("[]", encoding="utf-8")
    (output_dir / f"{doc_id}_middle.json").write_text("{}", encoding="utf-8")
    (output_dir / f"{doc_id}.md").write_text("# debug\n", encoding="utf-8")


def test_run_local_mineru_uses_fixed_command_env_and_force_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "paper1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "data" / "intermediate" / "mineru" / "paper1"
    stale_file = output_dir / "stale.txt"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("old", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, capture_output, text, env, timeout):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = env
        captured["timeout"] = timeout
        assert not stale_file.exists()
        _write_outputs(Path(command[command.index("-o") + 1]), "paper1")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner_module, "get_settings", _settings)
    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    artifacts = run_local_mineru(doc_id="paper1", pdf_path=pdf_path, device="cuda:0", force=True, root=tmp_path)

    assert captured["command"] == [
        "mineru-bin",
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-b",
        "pipeline",
        "-m",
        "auto",
        "-d",
        "cuda:0",
        "-t",
        "true",
        "-f",
        "true",
    ]
    assert captured["timeout"] == 12
    env = captured["env"]
    assert env["MINERU_DEVICE_MODE"] == "cuda:0"
    assert env["MINERU_TABLE_ENABLE"] == "true"
    assert env["MINERU_FORMULA_ENABLE"] == "true"
    assert env["MINERU_MODEL_SOURCE"] == "huggingface"
    assert env["MINERU_TOOLS_CONFIG_JSON"].endswith("data/model_cache/mineru/mineru.json")
    assert env["HF_HOME"].endswith("data/model_cache/huggingface")
    assert env["HUGGINGFACE_HUB_CACHE"].endswith("data/model_cache/huggingface/hub")
    assert env["TRANSFORMERS_CACHE"].endswith("data/model_cache/huggingface/transformers")
    assert env["SENTENCE_TRANSFORMERS_HOME"].endswith("data/model_cache/sentence_transformers")
    assert env["TORCH_HOME"].endswith("data/model_cache/torch")
    assert artifacts.status == "ok"
    assert artifacts.output_dir == str(output_dir)
    assert artifacts.content_list_path.endswith("paper1_content_list.json")
    assert artifacts.middle_json_path.endswith("paper1_middle.json")
    assert artifacts.markdown_path.endswith("paper1.md")
    assert artifacts.raw_images_dir.endswith("images")


def test_run_local_mineru_maps_timeout_to_pipeline_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "paper1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd=["mineru"], timeout=12)

    monkeypatch.setattr(runner_module, "get_settings", _settings)
    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    with pytest.raises(MinerUProcessError, match="timed out") as exc_info:
        run_local_mineru(doc_id="paper1", pdf_path=pdf_path, root=tmp_path)

    assert exc_info.value.error_type == "mineru_timeout"
    assert exc_info.value.context["doc_id"] == "paper1"


def test_run_local_mineru_maps_nonzero_exit_and_missing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "paper1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(runner_module, "get_settings", _settings)

    def fake_nonzero_run(command, capture_output, text, env, timeout):  # noqa: ANN001
        return SimpleNamespace(returncode=7, stdout="", stderr="mineru failed")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_nonzero_run)
    with pytest.raises(MinerUProcessError, match="mineru failed") as nonzero_exc:
        run_local_mineru(doc_id="paper1", pdf_path=pdf_path, root=tmp_path)
    assert nonzero_exc.value.error_type == "mineru_exit_nonzero"

    def fake_missing_output_run(command, capture_output, text, env, timeout):  # noqa: ANN001
        output_dir = Path(command[command.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_missing_output_run)
    with pytest.raises(MinerUProcessError, match="Ingest stage output is incomplete") as missing_exc:
        run_local_mineru(doc_id="paper1", pdf_path=pdf_path, root=tmp_path)
    assert missing_exc.value.error_type == "mineru_output_missing"
