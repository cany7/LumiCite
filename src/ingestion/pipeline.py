from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.core.errors import DocumentFetchError, MinerUProcessError
from src.core.logging import get_logger, report_error
from src.core.paths import (
    assets_root,
    bm25_index_path,
    bm25_metadata_path,
    chunks_jsonl_path,
    embeddings_jsonl_path,
    faiss_index_path,
    faiss_metadata_path,
    faiss_text_data_path,
    find_project_root,
    manifest_path,
    mineru_output_dir,
)
from src.core.schemas import ChunkModel
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import FaissStore
from src.ingestion.chunker import write_chunks_jsonl
from src.ingestion.embedder import build_embedding_records, load_canonical_chunks_jsonl, write_embeddings_jsonl
from src.ingestion.manifest import Manifest, ManifestDecision
from src.ingestion.mineru_mapper import map_mineru_output
from src.ingestion.mineru_runner import run_local_mineru
from src.ingestion.sources import create_source

logger = get_logger(__name__)


def _emit_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class PlannedJob:
    doc_id: str
    pdf_path: Path
    decision: ManifestDecision


@dataclass(frozen=True)
class ProcessedDocument:
    doc_id: str
    content_hash: str
    file_size_bytes: int
    chunks: list[ChunkModel]
    error: str | None = None
    error_type: str | None = None


def _embedding_file_model(embeddings_path: Path) -> str:
    if not embeddings_path.exists():
        return ""
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            return str(payload.get("embedding_model", "")).strip()
    return ""


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _group_chunks_by_doc(chunks: list[ChunkModel]) -> dict[str, list[ChunkModel]]:
    grouped: dict[str, list[ChunkModel]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped


def _remove_doc_artifacts(root: Path, doc_id: str, *, remove_pdf: bool = False) -> None:
    shutil.rmtree(mineru_output_dir(doc_id, root), ignore_errors=True)
    shutil.rmtree(assets_root(root) / doc_id, ignore_errors=True)
    if remove_pdf:
        pdf_path = root / "data" / "pdfs" / f"{doc_id}.pdf"
        if pdf_path.exists():
            pdf_path.unlink()


def _clear_index_artifacts(root: Path) -> None:
    paths = [
        faiss_index_path(root),
        faiss_metadata_path(root),
        faiss_text_data_path(root),
        bm25_index_path(root),
        bm25_metadata_path(root),
    ]
    for path in paths:
        if path.exists():
            path.unlink()


def _build_search_indices(
    *,
    chunks_path: Path,
    embeddings_path: Path,
    embedding_model: str,
) -> dict[str, Any]:
    faiss_store = FaissStore(embeddings_path=embeddings_path, embedding_model=embedding_model)
    faiss_store.build(embeddings_path)

    bm25_index = BM25Index(chunks_path=chunks_path)
    bm25_index.build()

    return {
        "faiss_index": str(faiss_store.index_path),
        "faiss_metadata": str(faiss_store.metadata_path),
        "text_data": str(faiss_store.text_data_path),
        "vectors": int(faiss_store.index.ntotal) if faiss_store.index is not None else 0,
        "bm25_index": str(bm25_index.index_path),
        "bm25_metadata": str(bm25_index.metadata_path),
        "bm25_rows": len(bm25_index.records),
    }


def _process_document(
    root: Path,
    doc_id: str,
    pdf_path: Path,
    decision: ManifestDecision,
    device: str,
    llm_backend: str | None,
) -> ProcessedDocument:
    _remove_doc_artifacts(root, doc_id)
    _emit_progress(f"Ingesting {doc_id}...")
    response = run_local_mineru(
        doc_id=doc_id,
        pdf_path=pdf_path,
        device=device,
        force=True,
        root=root,
    )
    _emit_progress(f"Processing extracted content for {doc_id}...")
    chunks = map_mineru_output(
        doc_id=doc_id,
        content_list_path=response.content_list_path,
        middle_json_path=response.middle_json_path,
        raw_images_dir=response.raw_images_dir,
        output_dir=response.output_dir,
        root=root,
        llm_backend=llm_backend,
    )
    return ProcessedDocument(
        doc_id=doc_id,
        content_hash=decision.content_hash,
        file_size_bytes=decision.file_size_bytes,
        chunks=chunks,
    )


def _run_jobs(
    root: Path,
    jobs: list[PlannedJob],
    device: str,
    llm_backend: str | None,
) -> list[ProcessedDocument]:
    if not jobs:
        return []

    results: list[ProcessedDocument] = []
    total_jobs = len(jobs)
    _emit_progress(f"Processing {total_jobs} PDFs sequentially...")
    for index, job in enumerate(jobs, start=1):
        _emit_progress(f"[{index}/{total_jobs}] Ingesting {job.doc_id}...")
        try:
            results.append(_process_document(root, job.doc_id, job.pdf_path, job.decision, device, llm_backend))
            _emit_progress(f"[{index}/{total_jobs}] Finished {job.doc_id}")
        except Exception as exc:
            _remove_doc_artifacts(root, job.doc_id)
            error_type = exc.error_type if isinstance(exc, MinerUProcessError) else "ingest_failed"
            report_error(
                logger,
                "ingest_failed",
                f"Ingest failed for {job.doc_id} ({error_type})",
                doc_id=job.doc_id,
                error_type=error_type,
                detail=str(exc),
            )
            results.append(
                ProcessedDocument(
                    doc_id=job.doc_id,
                    content_hash=job.decision.content_hash,
                    file_size_bytes=job.decision.file_size_bytes,
                    chunks=[],
                    error=str(exc),
                    error_type=error_type,
                )
            )
            _emit_progress(f"[{index}/{total_jobs}] Failed {job.doc_id}: {exc}")

    return results


def run_ingest(
    *,
    source: str,
    path: Path | None = None,
    device: str = "cpu",
    llm_backend: str | None = None,
    rebuild_index: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    root = find_project_root()
    pdf_dir = root / "data" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = chunks_jsonl_path(root)
    embeddings_path = embeddings_jsonl_path(root)

    source_impl = create_source(source, path)
    documents = source_impl.discover()
    snapshot_doc_ids = {document.doc_id for document in documents}
    _emit_progress(f"Discovered {len(documents)} documents from {source}.")

    manifest = Manifest(manifest_path(root))
    existing_chunks = load_canonical_chunks_jsonl(chunks_path)
    chunks_by_doc = _group_chunks_by_doc(existing_chunks)
    existing_doc_ids = set(chunks_by_doc) | set(manifest.entries)
    stale_doc_ids = existing_doc_ids - snapshot_doc_ids

    planned_actions: list[dict[str, Any]] = []
    jobs: list[PlannedJob] = []
    fetch_failures: list[str] = []
    skipped_docs = 0

    for stale_doc_id in sorted(stale_doc_ids):
        planned_actions.append({"doc_id": stale_doc_id, "action": "prune"})

    if source in {"url_csv", "url_list"} and documents:
        _emit_progress(f"Downloading PDFs from {source} into {pdf_dir} before ingest...")

    for index, document in enumerate(documents, start=1):
        try:
            if source in {"url_csv", "url_list"}:
                target_path = document.candidate_path(pdf_dir)
                if target_path.exists():
                    _emit_progress(f"[{index}/{len(documents)}] Reusing existing PDF for {document.doc_id}")
                else:
                    _emit_progress(f"[{index}/{len(documents)}] Downloading PDF for {document.doc_id}...")
            pdf_path = source_impl.fetch(document, pdf_dir)
            decision = manifest.should_process(
                document.doc_id,
                pdf_path,
                retry_failed_only=retry_failed,
            )
            if document.doc_id not in chunks_by_doc and decision.action == "skipped":
                decision = ManifestDecision(True, "missing_chunks", decision.content_hash, decision.file_size_bytes)
            planned_actions.append(
                {
                    "doc_id": document.doc_id,
                    "action": decision.action,
                    "pdf_path": str(pdf_path),
                }
            )
            if decision.should_process:
                jobs.append(PlannedJob(document.doc_id, pdf_path, decision))
            else:
                skipped_docs += 1
        except Exception as exc:
            fetch_failures.append(document.doc_id)
            error_type = exc.error_type if isinstance(exc, DocumentFetchError) else "fetch_failed"
            planned_actions.append(
                {
                    "doc_id": document.doc_id,
                    "action": "fetch_failed",
                    "error_type": error_type,
                    "error": str(exc),
                }
            )

    if source in {"url_csv", "url_list"} and documents:
        _emit_progress("Download phase finished. Starting ingest planning...")

    if dry_run:
        return {
            "event": "parse_dry_run",
            "source": source,
            "path": str(path) if path else "",
            "device": device,
            "retry_failed": retry_failed,
            "rebuild_index": rebuild_index,
            "discovered": len(documents),
            "planned_actions": planned_actions,
        }

    corpus_changed = False
    for stale_doc_id in stale_doc_ids:
        chunks_by_doc.pop(stale_doc_id, None)
        manifest.remove(stale_doc_id)
        _remove_doc_artifacts(root, stale_doc_id, remove_pdf=True)
        corpus_changed = True

    reparsed_doc_ids = sorted({job.doc_id for job in jobs if job.doc_id in chunks_by_doc})
    for reparsed_doc_id in reparsed_doc_ids:
        chunks_by_doc.pop(reparsed_doc_id, None)
        corpus_changed = True

    _emit_progress(f"Planned {len(jobs)} PDFs to process, {skipped_docs} to skip, {len(stale_doc_ids)} to prune.")
    results = _run_jobs(root, jobs, device, llm_backend or getattr(settings, "llm_backend", "api"))

    processed_doc_ids: set[str] = set()
    failed_doc_ids: list[str] = list(fetch_failures)
    for result in results:
        if result.error is not None:
            chunks_by_doc.pop(result.doc_id, None)
            failed_doc_ids.append(result.doc_id)
            manifest.set_failed(
                result.doc_id,
                content_hash=result.content_hash,
                file_size_bytes=result.file_size_bytes,
                embedding_model=settings.embedding_model,
                error_message=f"{result.error_type or 'ingest_failed'}: {result.error}",
            )
            continue

        chunks_by_doc[result.doc_id] = result.chunks
        manifest.set_complete(
            result.doc_id,
            content_hash=result.content_hash,
            file_size_bytes=result.file_size_bytes,
            num_chunks=len(result.chunks),
            embedding_model=settings.embedding_model,
        )
        processed_doc_ids.add(result.doc_id)
        corpus_changed = True

    all_chunks = [chunk for doc_id in sorted(chunks_by_doc) for chunk in chunks_by_doc[doc_id]]
    _emit_progress(f"Writing {len(all_chunks)} chunks to metadata store...")
    write_chunks_jsonl(all_chunks, chunks_path)

    embedding_model_changed = _embedding_file_model(embeddings_path) not in ("", settings.embedding_model)
    embeddings_need_rebuild = rebuild_index or corpus_changed or embedding_model_changed or not embeddings_path.exists()

    embedded_records = []
    if embeddings_need_rebuild:
        _emit_progress(f"Building embeddings for {len(all_chunks)} chunks...")
        embedded_records = build_embedding_records(
            all_chunks,
            embedding_model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        _emit_progress("Writing embedding records...")
        write_embeddings_jsonl(embedded_records, embeddings_path)
        manifest.update_embeddings(settings.embedding_model)

    index_info: dict[str, Any]
    indices_rebuilt = rebuild_index or corpus_changed or embeddings_need_rebuild
    if not all_chunks:
        _emit_progress("No chunks available. Clearing search indices...")
        _clear_index_artifacts(root)
        index_info = {
            "faiss_index": str(faiss_index_path(root)),
            "faiss_metadata": str(faiss_metadata_path(root)),
            "text_data": str(faiss_text_data_path(root)),
            "vectors": 0,
            "bm25_index": str(bm25_index_path(root)),
            "bm25_metadata": str(bm25_metadata_path(root)),
            "bm25_rows": 0,
        }
    else:
        faiss_store = FaissStore(embeddings_path=embeddings_path, embedding_model=settings.embedding_model)
        bm25_index = BM25Index(chunks_path=chunks_path)
        if not indices_rebuilt:
            try:
                indices_rebuilt = not (faiss_store.load() and bm25_index.load())
            except RuntimeError:
                indices_rebuilt = True

        if indices_rebuilt:
            _emit_progress("Rebuilding FAISS and BM25 indices...")
            index_info = _build_search_indices(
                chunks_path=chunks_path,
                embeddings_path=embeddings_path,
                embedding_model=settings.embedding_model,
            )
        else:
            _emit_progress("Reusing existing FAISS and BM25 indices...")
            index_info = {
                "faiss_index": str(faiss_store.index_path),
                "faiss_metadata": str(faiss_store.metadata_path),
                "text_data": str(faiss_store.text_data_path),
                "vectors": len(faiss_store.text_data),
                "bm25_index": str(bm25_index.index_path),
                "bm25_metadata": str(bm25_index.metadata_path),
                "bm25_rows": len(bm25_index.records),
            }

    _emit_progress("Saving manifest...")
    manifest.save()
    _emit_progress("Parse pipeline finished.")

    return {
        "event": "parse_summary",
        "status": "ok" if not failed_doc_ids else "partial",
        "source": source,
        "device": device,
        "processed_pdfs": len(processed_doc_ids),
        "failed_pdfs": len(failed_doc_ids),
        "skipped_pdfs": skipped_docs,
        "pruned_pdfs": len(stale_doc_ids),
        "chunks_written": len(all_chunks),
        "embeddings_written": len(embedded_records) if embeddings_need_rebuild else _count_jsonl_rows(embeddings_path),
        "rebuild_index": rebuild_index,
        "indices_rebuilt": indices_rebuilt,
        **index_info,
    }
