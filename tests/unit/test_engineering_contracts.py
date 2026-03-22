from __future__ import annotations

from pathlib import Path

import yaml


def _src_files() -> list[Path]:
    return sorted(Path("src").rglob("*.py"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_no_raw_print_calls_in_src() -> None:
    offenders: list[str] = []

    for path in _src_files():
        text = path.read_text(encoding="utf-8")
        if "print(" in text:
            offenders.append(str(path))

    assert offenders == []


def test_main_has_no_import_star() -> None:
    text = Path("src/main.py").read_text(encoding="utf-8")

    assert " import *" not in text


def test_docker_compose_keeps_optional_ollama_and_removes_mineru_parser() -> None:
    compose = _load_yaml(Path("docker-compose.yml"))
    services = compose["services"]
    api = services["api"]
    ollama = services["ollama"]

    assert "mineru-parser" not in services
    assert "depends_on" not in api
    assert "healthcheck" in api
    assert "healthcheck" in ollama
    assert ollama["build"]["dockerfile"] == "Dockerfile.ollama"
    assert ollama["environment"]["RAG_OLLAMA_MODEL"] == "${RAG_OLLAMA_MODEL:-qwen3.5:4b}"
    assert "/api/v1/health" in api["healthcheck"]["test"][-1]
    assert ollama["healthcheck"]["test"] == [
        "CMD-SHELL",
        "ollama list | grep -q '^${RAG_OLLAMA_MODEL:-qwen3.5:4b}[[:space:]]'",
    ]


def test_ollama_dockerfile_uses_bootstrap_entrypoint() -> None:
    text = Path("Dockerfile.ollama").read_text(encoding="utf-8")

    assert "FROM ollama/ollama:latest" in text
    assert "docker/ollama-entrypoint.sh" in text
    assert 'ENTRYPOINT ["/usr/local/bin/ollama-entrypoint.sh"]' in text
