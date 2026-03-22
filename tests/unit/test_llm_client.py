from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.generation.llm_client as llm_client_module


def test_ollama_status_requires_requested_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"models": [{"name": "qwen3.5:4b"}]}

    monkeypatch.setattr(llm_client_module.requests, "get", lambda *args, **kwargs: FakeResponse())

    assert llm_client_module._ollama_status("http://ollama.local:11434", "qwen3.5:4b") == (True, None)
    assert llm_client_module._ollama_status("http://ollama.local:11434", "qwen3:8b") == (
        False,
        "ollama_model_missing",
    )


def test_ensure_ollama_ready_waits_for_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = iter(
        [
            (False, "ollama_unavailable"),
            (False, "ollama_model_missing"),
            (True, None),
        ]
    )
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(
        llm_client_module,
        "get_settings",
        lambda: SimpleNamespace(
            ollama_base_url="http://ollama.local:11434",
            ollama_model="qwen3.5:4b",
            ollama_startup_timeout_seconds=5,
            ollama_poll_interval_seconds=0.0,
        ),
    )
    monkeypatch.setattr(llm_client_module, "find_project_root", lambda: "/repo")
    monkeypatch.setattr(
        llm_client_module,
        "_ollama_status",
        lambda base_url, model_name=None: next(statuses),
    )
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        llm_client_module.subprocess,
        "run",
        lambda command, **kwargs: (
            subprocess_calls.append(command) or SimpleNamespace(returncode=0, stderr="", stdout="")
        ),
    )

    result = llm_client_module.ensure_ollama_ready()

    assert result == "started"
    assert subprocess_calls == [["docker", "compose", "up", "-d", "ollama"]]
