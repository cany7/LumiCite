from __future__ import annotations

import shutil
from pathlib import Path

from src.core.logging import get_logger
from src.core.paths import doc_assets_dir, find_project_root

logger = get_logger(__name__)


def resolve_raw_asset_path(
    raw_asset_ref: str,
    *,
    output_dir: Path | None = None,
    raw_images_dir: Path | None = None,
) -> Path | None:
    reference = str(raw_asset_ref or "").strip()
    if not reference:
        return None

    raw_path = Path(reference)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    if output_dir is not None:
        candidates.append(output_dir / raw_path)
        candidates.append(output_dir / "images" / raw_path.name)
    if raw_images_dir is not None:
        candidates.append(raw_images_dir / raw_path.name)
        candidates.append(raw_images_dir / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def copy_asset_to_canonical(
    doc_id: str,
    chunk_id: str,
    raw_asset_path: Path | None,
    *,
    root: Path | None = None,
) -> str:
    if raw_asset_path is None or not raw_asset_path.exists():
        return ""

    project_root = root or find_project_root()
    destination_dir = doc_assets_dir(doc_id, project_root)
    extension = raw_asset_path.suffix.lower() or ".png"
    destination = destination_dir / f"{chunk_id}{extension}"

    if raw_asset_path.resolve() != destination.resolve():
        shutil.copy2(raw_asset_path, destination)

    relative = destination.relative_to(project_root)
    logger.info("visual_asset_copied", doc_id=doc_id, chunk_id=chunk_id, asset_path=relative.as_posix())
    return relative.as_posix()
