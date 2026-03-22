from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.config.settings import get_settings
from src.core.errors import DependencyError
from src.core.logging import get_logger
from src.core.paths import (
    huggingface_cache_dir,
    huggingface_hub_cache_dir,
    model_cache_dir,
    mineru_config_path,
    mineru_cache_dir,
    mineru_ready_marker_path,
    sentence_transformers_cache_dir,
    torch_cache_dir,
    transformers_cache_dir,
)

logger = get_logger(__name__)


def runtime_cache_env(root: Path | None = None) -> dict[str, str]:
    cache_root = model_cache_dir(root)
    return {
        "HF_HOME": str(huggingface_cache_dir(root)),
        "HUGGINGFACE_HUB_CACHE": str(huggingface_hub_cache_dir(root)),
        "TRANSFORMERS_CACHE": str(transformers_cache_dir(root)),
        "SENTENCE_TRANSFORMERS_HOME": str(sentence_transformers_cache_dir(root)),
        "TORCH_HOME": str(torch_cache_dir(root)),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "MPLCONFIGDIR": str(cache_root / "matplotlib"),
        "YOLO_CONFIG_DIR": str(cache_root / "ultralytics"),
    }


def configure_runtime_cache_environment(root: Path | None = None) -> dict[str, str]:
    env = runtime_cache_env(root)
    for key, value in env.items():
        os.environ[key] = value
        if key.endswith("_HOME") or key.endswith("_DIR") or key == "XDG_CACHE_HOME":
            Path(value).mkdir(parents=True, exist_ok=True)
    mineru_cache_dir(root)
    return env


def _load_embedding_model(model_name: str, *, root: Path | None = None, local_files_only: bool) -> None:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(
        model_name,
        cache_folder=str(sentence_transformers_cache_dir(root)),
        local_files_only=local_files_only,
    )


def _load_reranker_model(model_name: str, *, root: Path | None = None, local_files_only: bool) -> None:
    from sentence_transformers import CrossEncoder

    CrossEncoder(
        model_name,
        cache_folder=str(sentence_transformers_cache_dir(root)),
        local_files_only=local_files_only,
    )


def _ensure_model_available(
    *,
    label: str,
    model_name: str,
    loader,
    root: Path | None = None,
) -> None:
    configure_runtime_cache_environment(root)
    try:
        loader(model_name, root=root, local_files_only=True)
        logger.info("model_cache_hit", model_type=label, model=model_name)
        return
    except Exception:
        logger.info("model_cache_miss", model_type=label, model=model_name)

    try:
        loader(model_name, root=root, local_files_only=False)
        logger.info("model_downloaded", model_type=label, model=model_name)
    except Exception as exc:
        raise DependencyError(
            error_type=f"{label}_download_failed",
            message=f"Failed to download {label} model: {model_name}",
            retryable=False,
            context={"model": model_name, "detail": str(exc)},
        ) from exc


def ensure_embedding_model_available(model_name: str, *, root: Path | None = None) -> None:
    _ensure_model_available(
        label="embedding_model",
        model_name=model_name,
        loader=_load_embedding_model,
        root=root,
    )


def ensure_reranker_model_available(model_name: str, *, root: Path | None = None) -> None:
    _ensure_model_available(
        label="reranker_model",
        model_name=model_name,
        loader=_load_reranker_model,
        root=root,
    )


def _mineru_runtime_env(root: Path | None = None) -> dict[str, str]:
    return {
        **os.environ,
        **configure_runtime_cache_environment(root),
        "MINERU_TOOLS_CONFIG_JSON": str(mineru_config_path(root)),
    }


def _download_mineru_models(*, root: Path | None = None) -> None:
    env = _mineru_runtime_env(root)
    command = shutil.which("mineru-models-download")
    if command is None:
        raise DependencyError(
            error_type="mineru_download_command_missing",
            message="MinerU is not installed. Run uv sync first to install project dependencies.",
            retryable=False,
        )

    try:
        result = subprocess.run(
            [command, "-s", "huggingface", "-m", "all"],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
    except Exception as exc:
        raise DependencyError(
            error_type="mineru_model_download_failed",
            message="Failed to download MinerU models",
            retryable=False,
            context={"detail": str(exc)},
        ) from exc

    if result.returncode != 0:
        raise DependencyError(
            error_type="mineru_model_download_failed",
            message="Failed to download MinerU models",
            retryable=False,
            context={"detail": result.stderr.strip() or result.stdout.strip() or "mineru-models-download failed"},
        )

    marker = mineru_ready_marker_path(root)
    marker.write_text("ready\n", encoding="utf-8")
    logger.info("mineru_models_downloaded", cache_dir=str(mineru_cache_dir(root)))


def _mineru_models_ready(root: Path | None = None) -> bool:
    return mineru_ready_marker_path(root).exists()


def ensure_mineru_runtime_available(command: str, *, root: Path | None = None) -> None:
    if not _mineru_models_ready(root):
        _download_mineru_models(root=root)

    env = _mineru_runtime_env(root)
    resolved = shutil.which(command)
    if resolved is None:
        raise DependencyError(
            error_type="mineru_command_missing",
            message=f"MinerU command is not available: {command}",
            retryable=False,
            context={"command": command},
        )

    try:
        result = subprocess.run(
            [resolved, "--help"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
    except Exception as exc:
        raise DependencyError(
            error_type="mineru_probe_failed",
            message=f"Failed to initialize MinerU runtime: {command}",
            retryable=False,
            context={"command": command, "detail": str(exc)},
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "mineru help probe failed"
        raise DependencyError(
            error_type="mineru_probe_failed",
            message=f"Failed to initialize MinerU runtime: {command}",
            retryable=False,
            context={"command": command, "detail": detail},
        )

    logger.info("mineru_runtime_ready", command=command, cache_dir=str(mineru_cache_dir(root)))


def ensure_parse_runtime_dependencies(*, root: Path | None = None) -> dict[str, str]:
    settings = get_settings()
    env = configure_runtime_cache_environment(root)
    console = Console(stderr=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=24),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        embedding_task = progress.add_task("Embedding model", total=1)
        reranker_task = progress.add_task("Reranker model", total=1)
        mineru_task = progress.add_task("MinerU models", total=1)

        progress.update(embedding_task, description="Embedding model")
        ensure_embedding_model_available(settings.embedding_model, root=root)
        progress.advance(embedding_task)
        progress.update(embedding_task, description="Embedding model ready")

        progress.update(reranker_task, description="Reranker model")
        ensure_reranker_model_available(settings.reranker_model, root=root)
        progress.advance(reranker_task)
        progress.update(reranker_task, description="Reranker model ready")

        progress.update(mineru_task, description="MinerU models")
        ensure_mineru_runtime_available(settings.mineru_command, root=root)
        progress.advance(mineru_task)
        progress.update(mineru_task, description="MinerU models ready")
    return env
