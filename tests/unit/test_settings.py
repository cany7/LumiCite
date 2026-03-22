from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.settings import Settings


def test_settings_loads_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RAG_RETRIEVAL_MODE=sparse\n"
        "RAG_LOG_FORMAT=json\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.retrieval_mode == "sparse"
    assert settings.log_format == "json"


def test_settings_ignores_unknown_env_keys(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RAG_RETRIEVAL_MODE=hybrid\n"
        "RAG_UNKNOWN_KEY=ignored\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.retrieval_mode == "hybrid"


def test_settings_validates_literal_fields(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_RETRIEVAL_MODE=invalid\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)
