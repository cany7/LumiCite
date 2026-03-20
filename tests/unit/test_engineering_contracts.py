from __future__ import annotations

from pathlib import Path

import yaml


def _src_files() -> list[Path]:
    return sorted(Path("src").rglob("*.py"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_no_raw_print_calls_in_src():
    offenders: list[str] = []

    for path in _src_files():
        text = path.read_text(encoding="utf-8")
        if "print(" in text:
            offenders.append(str(path))

    assert offenders == []


def test_main_has_no_import_star():
    main_path = Path("src/main.py")
    text = main_path.read_text(encoding="utf-8")

    assert " import *" not in text


def test_ci_workflow_uses_frozen_sync():
    workflow = _load_yaml(Path(".github/workflows/ci.yml"))
    steps = workflow["jobs"]["checks"]["steps"]
    run_steps = [step.get("run", "") for step in steps if isinstance(step, dict)]

    assert "uv sync --frozen" in run_steps


def test_docker_compose_waits_for_healthy_ollama_and_declares_healthchecks():
    compose = _load_yaml(Path("docker-compose.yml"))
    api = compose["services"]["api"]
    ollama = compose["services"]["ollama"]

    assert api["depends_on"]["ollama"]["condition"] == "service_healthy"
    assert "healthcheck" in api
    assert "healthcheck" in ollama
    assert "/api/v1/health" in api["healthcheck"]["test"][-1]
    assert ollama["healthcheck"]["test"] == ["CMD", "ollama", "list"]
