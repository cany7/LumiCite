from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import time

from src.config.settings import get_settings
from src.core.errors import DocumentFetchError
from src.core.logging import get_logger
from src.core.logging import report_error
from src.ingestion.sources.base import DocumentMeta

logger = get_logger(__name__)


def filename_from_url(url: str, index: int) -> str:
    parsed = urlparse(url)
    filename = Path(unquote(parsed.path)).name.strip()
    if not filename:
        filename = f"url_{index}.pdf"
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return filename


def build_url_document(
    *,
    url: str,
    index: int,
    source_type: str,
    metadata: dict[str, str] | None = None,
) -> DocumentMeta:
    filename = filename_from_url(url, index)
    return DocumentMeta(
        doc_id=Path(filename).stem,
        source_type=source_type,
        filename=filename,
        url=url,
        metadata=dict(metadata or {}),
    )


def dedupe_documents(documents: list[DocumentMeta]) -> list[DocumentMeta]:
    deduped: list[DocumentMeta] = []
    seen_doc_ids: set[str] = set()
    for document in documents:
        if document.doc_id in seen_doc_ids:
            logger.warning("duplicate_doc_id_skipped", doc_id=document.doc_id, source_type=document.source_type)
            continue
        seen_doc_ids.add(document.doc_id)
        deduped.append(document)
    return deduped


def _classify_request_exception(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, requests.Timeout):
        return "timeout", True
    if isinstance(exc, requests.ConnectionError):
        return "connection_error", True
    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 404:
            return "not_found", False
        if status_code == 403:
            return "forbidden", False
        if status_code is not None and 500 <= status_code <= 599:
            return "temporary_http_error", True
        return "request_error", False
    if isinstance(exc, requests.RequestException):
        return "request_error", False
    return "request_error", False


def fetch_url_document(doc: DocumentMeta, dest_dir: Path, *, timeout: int | None = None) -> Path:
    if not doc.url:
        raise FileNotFoundError(f"No URL available for document: {doc.doc_id}")

    settings = get_settings()
    effective_timeout = timeout or settings.download_timeout_seconds
    max_attempts = max(1, settings.request_retry_attempts)
    retry_delay = max(0.0, settings.request_retry_delay_seconds)

    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / doc.filename
    if target_path.exists():
        return target_path

    last_error_type = "request_error"
    last_message = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(doc.url, timeout=effective_timeout)
            response.raise_for_status()
            target_path.write_bytes(response.content)
            logger.info("url_document_fetched", doc_id=doc.doc_id, path=str(target_path), source_type=doc.source_type)
            return target_path
        except Exception as exc:
            error_type, retryable = _classify_request_exception(exc)
            last_error_type = error_type
            last_message = str(exc)
            should_retry = retryable and attempt < max_attempts
            if should_retry:
                time.sleep(retry_delay)
                continue

            report_error(
                logger,
                "fetch_failed",
                f"PDF download failed for {doc.doc_id} ({error_type})",
                doc_id=doc.doc_id,
                url=doc.url,
                error_type=error_type,
                detail=last_message,
            )
            raise DocumentFetchError(
                error_type=error_type,
                message=f"Failed to fetch {doc.doc_id}: {error_type}",
                retryable=False,
                context={"doc_id": doc.doc_id, "url": doc.url, "detail": last_message},
            ) from exc

    raise DocumentFetchError(
        error_type=last_error_type,
        message=f"Failed to fetch {doc.doc_id}: {last_error_type}",
        retryable=False,
        context={"doc_id": doc.doc_id, "url": doc.url, "detail": last_message},
    )
