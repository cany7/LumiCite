from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineError(Exception):
    error_type: str
    message: str
    retryable: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class DocumentFetchError(PipelineError):
    pass


class MinerUProcessError(PipelineError):
    pass


class VisualInferenceError(PipelineError):
    pass


class GenerationError(PipelineError):
    pass


class OllamaReadyError(PipelineError):
    pass


class DependencyError(PipelineError):
    pass
