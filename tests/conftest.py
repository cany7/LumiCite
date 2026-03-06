from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def json_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "JSON"
    path.mkdir(parents=True, exist_ok=True)
    return path
