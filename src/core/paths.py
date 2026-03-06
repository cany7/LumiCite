from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root by searching upwards for src/ and data/."""
    path = (start or Path(__file__)).resolve()
    if path.is_file():
        path = path.parent

    while True:
        if (path / "src").exists() and (path / "data").exists():
            return path
        if path.parent == path:
            raise FileNotFoundError("Could not find project root containing src/ and data/.")
        path = path.parent
