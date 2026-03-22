from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import get_settings
from src.core.errors import MinerUProcessError
from src.core.model_assets import runtime_cache_env
from src.core.paths import mineru_config_path, mineru_output_dir, mineru_ready_marker_path

INGEST_OUTPUT_INCOMPLETE_MESSAGE = "Ingest stage output is incomplete"


@dataclass(frozen=True)
class MinerUArtifacts:
    status: str
    doc_id: str
    output_dir: str
    content_list_path: str
    middle_json_path: str
    markdown_path: str
    raw_images_dir: str


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _find_first_match(output_dir: Path, patterns: list[str], *, fallback: Path | None = None) -> Path:
    for pattern in patterns:
        matches = sorted(output_dir.rglob(pattern))
        if matches:
            return matches[0]
    if fallback is not None:
        return fallback
    raise MinerUProcessError(
        error_type="mineru_output_missing",
        message=INGEST_OUTPUT_INCOMPLETE_MESSAGE,
        context={"output_dir": str(output_dir)},
    )


def _locate_outputs(doc_id: str, output_dir: Path) -> MinerUArtifacts:
    content_list_path = _find_first_match(
        output_dir,
        [f"{doc_id}_content_list.json", "*content_list*.json"],
    )
    middle_json_path = _find_first_match(
        output_dir,
        [f"{doc_id}_middle.json", "*_middle.json", "*middle*.json"],
    )
    markdown_path = _find_first_match(
        output_dir,
        [f"{doc_id}.md", "*.md"],
        fallback=output_dir / f"{doc_id}.md",
    )
    raw_images_dir = _find_first_match(
        output_dir,
        ["images"],
        fallback=output_dir / "images",
    )
    expected_paths = [content_list_path, middle_json_path, markdown_path, raw_images_dir]
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise MinerUProcessError(
            error_type="mineru_output_missing",
            message=INGEST_OUTPUT_INCOMPLETE_MESSAGE,
            context={"doc_id": doc_id, "missing_paths": missing},
        )
    return MinerUArtifacts(
        status="ok",
        doc_id=doc_id,
        output_dir=str(output_dir),
        content_list_path=str(content_list_path),
        middle_json_path=str(middle_json_path),
        markdown_path=str(markdown_path),
        raw_images_dir=str(raw_images_dir),
    )


def _tail_output(text: str, *, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def run_local_mineru(
    *,
    doc_id: str,
    pdf_path: str | Path,
    device: str = "cpu",
    force: bool = True,
    root: Path | None = None,
) -> MinerUArtifacts:
    settings = get_settings()
    source_path = Path(pdf_path)
    output_dir = mineru_output_dir(doc_id, root)
    _prepare_output_dir(output_dir, force=force)

    env = os.environ.copy()
    env.update(runtime_cache_env(root))
    env["MINERU_TOOLS_CONFIG_JSON"] = str(mineru_config_path(root))
    env.update(
        {
            "MINERU_DEVICE_MODE": device,
            "MINERU_TABLE_ENABLE": "true",
            "MINERU_FORMULA_ENABLE": "true",
            "MINERU_MODEL_SOURCE": "local" if mineru_ready_marker_path(root).exists() else settings.mineru_model_source,
        }
    )
    command = [
        settings.mineru_command,
        "-p",
        str(source_path),
        "-o",
        str(output_dir),
        "-b",
        "pipeline",
        "-m",
        "auto",
        "-d",
        device,
        "-t",
        "true",
        "-f",
        "true",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=settings.mineru_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise MinerUProcessError(
            error_type="mineru_timeout",
            message=f"MinerU timed out for {doc_id}",
            context={
                "doc_id": doc_id,
                "pdf_path": str(source_path),
                "timeout_seconds": settings.mineru_timeout_seconds,
            },
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"mineru exited with code {result.returncode}"
        raise MinerUProcessError(
            error_type="mineru_exit_nonzero",
            message=detail,
            context={
                "doc_id": doc_id,
                "pdf_path": str(source_path),
                "returncode": result.returncode,
            },
        )

    try:
        return _locate_outputs(doc_id, output_dir)
    except MinerUProcessError as exc:
        detail = _tail_output(result.stderr) or _tail_output(result.stdout)
        if detail:
            raise MinerUProcessError(
                error_type=exc.error_type,
                message=f"{INGEST_OUTPUT_INCOMPLETE_MESSAGE}. MinerU output tail:\n{detail}",
                context={
                    **exc.context,
                    "doc_id": doc_id,
                    "pdf_path": str(source_path),
                },
            ) from exc
        raise
