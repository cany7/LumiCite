from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.core.logging import get_logger
from src.core.paths import find_project_root

logger = get_logger(__name__)


def load_metadata_df(path: Path | None = None) -> pd.DataFrame:
    root = find_project_root()
    meta_path = path or (root / "data" / "metadata" / "metadata.csv")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    df = pd.read_csv(meta_path, encoding="utf-8")
    logger.info("metadata_loaded", path=str(meta_path), rows=len(df))
    return df


def get_pdf_dir() -> Path:
    root = find_project_root()
    return root / "data" / "pdfs"


def get_PDF_paths() -> dict[str, str]:
    pdf_dir = get_pdf_dir()
    if not pdf_dir.exists():
        logger.warning("pdf_dir_missing", path=str(pdf_dir))
        return {}

    files = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    logger.info("pdfs_discovered", count=len(files), path=str(pdf_dir))
    return {p.stem: str(p) for p in files}


def download_pdfs(df: pd.DataFrame, target_dir: Path | None = None) -> list[Path]:
    pdf_dir = target_dir or get_pdf_dir()
    pdf_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths: list[Path] = []
    for _, row in df.iterrows():
        doc_id = str(row.get("id", "")).strip()
        url = str(row.get("url", "")).strip()
        if not doc_id or not url:
            continue

        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if pdf_path.exists():
            downloaded_paths.append(pdf_path)
            continue

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
            downloaded_paths.append(pdf_path)
            logger.info("pdf_downloaded", id=doc_id, path=str(pdf_path))
        except Exception as exc:
            logger.warning("pdf_download_failed", id=doc_id, error=str(exc))

    return sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])


def load_data() -> tuple[pd.DataFrame, list[Path]]:
    df = load_metadata_df()
    pdfs = download_pdfs(df)
    return df, pdfs
