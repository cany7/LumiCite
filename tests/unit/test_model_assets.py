from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.core.model_assets as assets_module
from src.core.errors import DependencyError


def test_runtime_cache_env_uses_project_model_cache(tmp_path: Path) -> None:
    env = assets_module.runtime_cache_env(tmp_path)

    assert env["HF_HOME"] == str(tmp_path / "data" / "model_cache" / "huggingface")
    assert env["HUGGINGFACE_HUB_CACHE"] == str(tmp_path / "data" / "model_cache" / "huggingface" / "hub")
    assert env["TRANSFORMERS_CACHE"] == str(tmp_path / "data" / "model_cache" / "huggingface" / "transformers")
    assert env["SENTENCE_TRANSFORMERS_HOME"] == str(tmp_path / "data" / "model_cache" / "sentence_transformers")
    assert env["TORCH_HOME"] == str(tmp_path / "data" / "model_cache" / "torch")


def test_ensure_embedding_model_downloads_when_local_cache_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, cache_folder: str | None = None, local_files_only: bool = False, **kwargs) -> None:  # noqa: ANN003
            calls.append(
                {
                    "model_name": model_name,
                    "cache_folder": cache_folder,
                    "local_files_only": local_files_only,
                }
            )
            if local_files_only:
                raise OSError("missing local files")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    assets_module.ensure_embedding_model_available("sentence-transformers/all-MiniLM-L6-v2", root=tmp_path)

    assert calls == [
        {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "cache_folder": str(tmp_path / "data" / "model_cache" / "sentence_transformers"),
            "local_files_only": True,
        },
        {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "cache_folder": str(tmp_path / "data" / "model_cache" / "sentence_transformers"),
            "local_files_only": False,
        },
    ]


def test_ensure_reranker_model_raises_clear_error_on_download_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCrossEncoder:
        def __init__(self, model_name: str, cache_folder: str | None = None, local_files_only: bool = False, **kwargs) -> None:  # noqa: ANN003
            raise RuntimeError("download failed")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(DependencyError, match="Failed to download reranker_model model") as exc_info:
        assets_module.ensure_reranker_model_available("cross-encoder/ms-marco-MiniLM-L-6-v2", root=tmp_path)

    assert exc_info.value.error_type == "reranker_model_download_failed"


def test_ensure_parse_runtime_dependencies_checks_all_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str]] = []

    monkeypatch.setattr(
        assets_module,
        "get_settings",
        lambda: SimpleNamespace(
            embedding_model="embedding-model",
            reranker_model="reranker-model",
            mineru_command="mineru-bin",
        ),
    )
    monkeypatch.setattr(
        assets_module,
        "ensure_embedding_model_available",
        lambda model_name, root=None: captured.append(("embedding", model_name)),
    )
    monkeypatch.setattr(
        assets_module,
        "ensure_reranker_model_available",
        lambda model_name, root=None: captured.append(("reranker", model_name)),
    )
    monkeypatch.setattr(
        assets_module,
        "ensure_mineru_runtime_available",
        lambda command, root=None: captured.append(("mineru", command)),
    )

    assets_module.ensure_parse_runtime_dependencies(root=tmp_path)

    assert captured == [
        ("embedding", "embedding-model"),
        ("reranker", "reranker-model"),
        ("mineru", "mineru-bin"),
    ]


def test_ensure_mineru_runtime_downloads_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(assets_module.shutil, "which", lambda name: f"/venv/bin/{name}")

    def fake_run(command, env, timeout, **kwargs):  # noqa: ANN001
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(assets_module.subprocess, "run", fake_run)

    assets_module.ensure_mineru_runtime_available("mineru", root=tmp_path)

    assert calls == [
        ["/venv/bin/mineru-models-download", "-s", "huggingface", "-m", "all"],
        ["/venv/bin/mineru", "--help"],
    ]
    assert (tmp_path / "data" / "model_cache" / "mineru" / ".models_ready").exists()


def test_ensure_mineru_runtime_reuses_downloaded_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    ready_marker = tmp_path / "data" / "model_cache" / "mineru" / ".models_ready"
    ready_marker.parent.mkdir(parents=True, exist_ok=True)
    ready_marker.write_text("ready\n", encoding="utf-8")

    monkeypatch.setattr(assets_module.shutil, "which", lambda name: f"/venv/bin/{name}")

    def fake_run(command, capture_output, text, env, timeout, **kwargs):  # noqa: ANN001
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(assets_module.subprocess, "run", fake_run)

    assets_module.ensure_mineru_runtime_available("mineru", root=tmp_path)

    assert calls == [["/venv/bin/mineru", "--help"]]
