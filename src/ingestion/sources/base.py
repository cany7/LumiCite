from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.paths import find_project_root


@dataclass(frozen=True)
class DocumentMeta:
    doc_id: str
    source_type: str
    filename: str
    local_path: Path | None = None
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def candidate_path(self, dest_dir: Path) -> Path:
        if self.local_path is not None:
            return self.local_path
        return dest_dir / self.filename


class BaseSource(ABC):
    def __init__(self, path: str | Path | None = None) -> None:
        self.root = find_project_root()
        self.path = self._resolve_path(path) if path is not None else None

    def _resolve_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        project_relative = self.root / path
        return project_relative if project_relative.exists() else path

    def _resolve_default_input_file(self, pattern: str, *, label: str) -> Path:
        search_dir = self.root / "data" / "pdfs"
        matches = sorted(path for path in search_dir.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"No {label} file found in {search_dir}")
        if len(matches) > 1:
            matched = ", ".join(path.name for path in matches)
            raise FileNotFoundError(
                f"Multiple {label} files found in {search_dir}: {matched}. "
                "Please pass --path to choose one."
            )
        return matches[0]

    @abstractmethod
    def discover(self) -> list[DocumentMeta]:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, doc: DocumentMeta, dest_dir: Path) -> Path:
        raise NotImplementedError
