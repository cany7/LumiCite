from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

import structlog

from src.config.settings import get_settings

_F = TypeVar("_F", bound=Callable[..., Any])
_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
    ]

    renderer: Any
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared, renderer],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _configured = True


def get_logger(name: str | None = None) -> Any:
    configure_logging()
    return structlog.get_logger(name)


def timed(name: str | None = None) -> Callable[[_F], _F]:
    def decorator(func: _F) -> _F:
        metric_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info("timed", operation=metric_name, latency_ms=round(latency_ms, 3))

        return wrapper  # type: ignore[return-value]

    return decorator
