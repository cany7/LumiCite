from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.core.logging as logging_module
from src.core.errors import DocumentFetchError, GenerationError, MinerUProcessError, OllamaReadyError, PipelineError
from src.core.paths import (
    bm25_index_path,
    chunks_jsonl_path,
    doc_assets_dir,
    embeddings_jsonl_path,
    find_project_root,
    manifest_path,
    mineru_output_dir,
    rag_log_path,
)


@pytest.fixture(autouse=True)
def reset_logging_state() -> None:
    logging.getLogger().handlers.clear()
    logging_module._configured = False
    yield
    logging.getLogger().handlers.clear()
    logging_module._configured = False


def test_find_project_root_and_canonical_paths_follow_repo_layout(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "src" / "pkg" / "module.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("# module\n", encoding="utf-8")
    (root / "data").mkdir()

    assert find_project_root(nested) == root
    assert chunks_jsonl_path(root) == root / "data" / "metadata" / "chunks" / "chunks.jsonl"
    assert embeddings_jsonl_path(root) == root / "data" / "metadata" / "embeddings" / "embeddings.jsonl"
    assert bm25_index_path(root) == root / "data" / "metadata" / "bm25" / "bm25_index.pkl"
    assert manifest_path(root) == root / "data" / "manifest.json"
    assert mineru_output_dir("paper1", root) == root / "data" / "intermediate" / "mineru" / "paper1"
    assert doc_assets_dir("paper1", root) == root / "data" / "assets" / "paper1"
    assert rag_log_path(root) == root / "rag.log"


def test_report_error_writes_rag_log_and_cli_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    log_path = tmp_path / "rag.log"

    monkeypatch.setattr(logging_module, "get_settings", lambda: SimpleNamespace(log_format="json"))
    monkeypatch.setattr(logging_module, "rag_log_path", lambda: log_path)

    logger = logging_module.get_logger("tests.core")
    logging_module.report_error(
        logger,
        "fetch_failed",
        "PDF download failed",
        error_type="timeout",
        doc_id="paper1",
    )

    captured = capsys.readouterr()
    log_text = log_path.read_text(encoding="utf-8")

    assert "PDF download failed" in captured.err
    assert "fetch_failed" in log_text
    assert "paper1" in log_text
    assert "timeout" in log_text


@pytest.mark.parametrize(
    ("error_cls", "error_type"),
    [
        (DocumentFetchError, "fetch_failed"),
        (MinerUProcessError, "mineru_exit_nonzero"),
        (GenerationError, "generation_request_error"),
        (OllamaReadyError, "ollama_unavailable"),
    ],
)
def test_pipeline_error_subclasses_preserve_contract_fields(error_cls: type[PipelineError], error_type: str) -> None:
    error = error_cls(
        error_type=error_type,
        message="boom",
        retryable=True,
        context={"doc_id": "paper1"},
    )

    assert str(error) == "boom"
    assert error.error_type == error_type
    assert error.retryable is True
    assert error.context == {"doc_id": "paper1"}

