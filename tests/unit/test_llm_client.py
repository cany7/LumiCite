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


def test_api_client_passes_reasoning_effort_via_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="ANSWER: 55%\nSUPPORTING_MATERIALS: evidence\nEXPLANATION: because\nCITED_CHUNK_IDS:\n- paper1_deadbeef"
                        )
                    )
                ]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    client = llm_client_module.APIClient(
        model="qwen/test",
        api_key="key",
        base_url="https://openrouter.ai/api/v1",
        reasoning_effort="low",
    )
    monkeypatch.setattr(client, "_load_client", lambda: FakeClient())

    text = client.generate("What happened?")

    assert "ANSWER: 55%" in text
    assert captured["extra_body"] == {"reasoning": {"effort": "low", "exclude": True}}


def test_api_client_omits_extra_body_when_reasoning_effort_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ANSWER: ok"))]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    client = llm_client_module.APIClient(
        model="qwen/test",
        api_key="key",
        base_url="https://openrouter.ai/api/v1",
    )
    monkeypatch.setattr(client, "_load_client", lambda: FakeClient())

    client.generate("What happened?")

    assert "extra_body" not in captured


def test_api_client_allows_custom_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="rewritten query"))]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    client = llm_client_module.APIClient(
        model="qwen/test",
        api_key="key",
        base_url="https://openrouter.ai/api/v1",
    )
    monkeypatch.setattr(client, "_load_client", lambda: FakeClient())

    text = client.generate("Question: test", system_prompt="Rewrite only")

    assert text == "rewritten query"
    assert captured["messages"][0]["content"] == "Rewrite only"
