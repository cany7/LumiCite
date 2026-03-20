from __future__ import annotations

from fastapi import FastAPI
from typer.testing import CliRunner

import src.main as main_module
from src.main import app

runner = CliRunner()


def test_serve_starts_uvicorn_with_app_instance(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(target, **kwargs):  # noqa: ANN001, ANN003
        captured["target"] = target
        captured["kwargs"] = kwargs

    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "8080"])

    assert result.exit_code == 0
    assert isinstance(captured["target"], FastAPI)
    assert captured["kwargs"] == {
        "host": "127.0.0.1",
        "port": 8080,
        "reload": False,
    }


def test_serve_reload_uses_app_factory(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(target, **kwargs):  # noqa: ANN001, ANN003
        captured["target"] = target
        captured["kwargs"] = kwargs

    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "8081", "--reload"])

    assert result.exit_code == 0
    assert captured["target"] == "src.api.app:create_app"
    assert captured["kwargs"] == {
        "factory": True,
        "host": "127.0.0.1",
        "port": 8081,
        "reload": True,
    }
