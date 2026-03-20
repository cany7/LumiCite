from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
import typer
import uvicorn
from fastapi import FastAPI

from src.config.settings import get_settings
from src.core.logging import get_logger
from src.core.paths import find_project_root
from src.core.schemas import EmbeddingRecord, TextChunk
from src.evaluation.evaluator import Evaluator
from src.generation.rag_pipeline import RAGConfig, RAGPipeline
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import FaissStore
from src.ingestion.manifest import Manifest
from src.ingestion.sources import create_source
from src.ingestion.sources.base import DocumentMeta
from src.ingestion.chunker import extract_pdf_chunks, write_chunks_jsonl
from src.ingestion.embedder import embed_chunks, load_canonical_chunks_jsonl, write_embeddings_jsonl
from src.ingestion.legacy_loader import load_legacy_chunks_json
from src.retrieval import get_retriever
from src.retrieval.reranker import Reranker

app = typer.Typer(help="RAG command line interface")
logger = get_logger(__name__)


def _now_utc() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _canonical_paths(root: Path) -> tuple[Path, Path]:
    json_dir = root / "data" / "JSON"
    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir / "chunks.jsonl", json_dir / "embeddings.jsonl"


def _embedding_file_model(embeddings_path: Path) -> str:
    if not embeddings_path.exists():
        return ""
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            return str(payload.get("embedding_model", ""))
    return ""


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _build_search_indices(
    chunks_path: Path,
    embeddings_path: Path,
    embedding_model: str,
) -> dict[str, object]:
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


def _retrieve_results(question: str, top_k: int, retrieval_mode: str, rerank: bool = False) -> list[dict]:
    retriever = get_retriever(retrieval_mode)
    fetch_k = top_k * 3 if rerank else top_k
    results = retriever.retrieve(question, fetch_k)
    if not rerank:
        return results[:top_k]
    reranker = Reranker()
    return reranker.rerank(question, results, top_k)


def _chunk_strategy() -> str:
    settings = get_settings()
    return f"hybrid_{settings.chunk_size}_{settings.chunk_overlap}"


def _load_existing_chunks(root: Path, chunks_path: Path) -> list[Any]:
    if chunks_path.exists():
        return load_canonical_chunks_jsonl(chunks_path)

    chunks: list[Any] = []
    legacy_sources = [root / "data" / "JSON" / "chunks.json", root / "data" / "JSON" / "alt_text.json"]
    for legacy in legacy_sources:
        if legacy.exists():
            chunks.extend(load_legacy_chunks_json(legacy))
    return chunks


def _group_chunks_by_doc(chunks: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped


def _seed_manifest_from_existing_corpus(
    manifest: Manifest,
    documents: list[DocumentMeta],
    chunks_by_doc: dict[str, list[Any]],
    pdf_dir: Path,
    *,
    chunk_strategy: str,
    embedding_model: str,
) -> int:
    if manifest.entries:
        return 0

    seeded = 0
    for doc in documents:
        existing_chunks = chunks_by_doc.get(doc.doc_id)
        if not existing_chunks:
            continue

        pdf_path = doc.local_path or doc.candidate_path(pdf_dir)
        if not pdf_path.exists():
            continue

        decision = manifest.should_process(
            doc.doc_id,
            pdf_path,
            chunk_strategy=chunk_strategy,
            embedding_model=embedding_model,
        )
        manifest.set_complete(
            doc.doc_id,
            content_hash=decision.content_hash,
            file_size_bytes=decision.file_size_bytes,
            chunk_strategy=chunk_strategy,
            num_chunks=len(existing_chunks),
            embedding_model=embedding_model,
        )
        seeded += 1

    if seeded:
        manifest.save()
    return seeded


def _records_for_embedding(chunks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": {
                "doc_id": chunk.doc_id,
                "page_number": chunk.page_number,
                "headings": chunk.headings,
                "source_file": chunk.source_file,
                "chunk_type": chunk.chunk_type.value,
            },
        }
        for chunk in chunks
    ]


def _extract_pdf_worker(pdf_path_str: str) -> tuple[str, list[Any], str | None]:
    simulate_ms = int(os.environ.get("RAG_INGEST_SIMULATE_MS", "0") or "0")
    if simulate_ms > 0:
        time.sleep(simulate_ms / 1000)
        pdf_path = Path(pdf_path_str)
        doc_id = pdf_path.stem or "simulated"
        return (
            pdf_path_str,
            [
                TextChunk(
                    chunk_id=f"{doc_id}_{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    text=f"Simulated chunk for {doc_id}",
                    source_file=pdf_path.name,
                )
            ],
            None,
        )

    try:
        chunks = extract_pdf_chunks(pdf_path_str)
        return pdf_path_str, chunks, None
    except Exception as exc:  # pragma: no cover - exercised via integration path
        return pdf_path_str, [], str(exc)


def _extract_chunks_for_paths(pdf_paths: list[Path], workers: int) -> tuple[dict[str, list[Any]], dict[str, str]]:
    if not pdf_paths:
        return {}, {}

    chunk_results: dict[str, list[Any]] = {}
    errors: dict[str, str] = {}
    if workers <= 1 or len(pdf_paths) == 1:
        for pdf_path in pdf_paths:
            path_str, chunks, error = _extract_pdf_worker(str(pdf_path))
            if error is not None:
                errors[path_str] = error
            else:
                chunk_results[path_str] = chunks
        return chunk_results, errors

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_extract_pdf_worker, str(pdf_path)) for pdf_path in pdf_paths]
        for future in as_completed(futures):
            path_str, chunks, error = future.result()
            if error is not None:
                errors[path_str] = error
            else:
                chunk_results[path_str] = chunks

    return chunk_results, errors


def _discover_pdfs(source: str, path: Path | None) -> list[Path]:
    root = find_project_root()

    if source == "metadata_csv":
        metadata_path = path or (root / "data" / "metadata" / "metadata.csv")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
        df = pd.read_csv(metadata_path)
        pdf_dir = root / "data" / "pdfs"
        return [pdf_dir / f"{str(doc_id)}.pdf" for doc_id in df["id"].tolist() if (pdf_dir / f"{str(doc_id)}.pdf").exists()]

    if source == "local_dir":
        pdf_dir = path or (root / "data" / "pdfs")
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        return sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])

    if source == "url_list":
        url_file = path
        if url_file is None or not url_file.exists():
            raise FileNotFoundError("URL list file is required for --source url_list")

        pdf_dir = root / "data" / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for idx, line in enumerate(url_file.read_text(encoding="utf-8").splitlines(), start=1):
            url = line.strip()
            if not url:
                continue
            parsed = urlparse(url)
            filename = Path(parsed.path).name or f"url_{idx}.pdf"
            if not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"
            out_path = pdf_dir / filename
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                out_path.write_bytes(response.content)
                downloaded.append(out_path)
            except Exception as exc:
                logger.warning("url_download_failed", url=url, error=str(exc))
        return downloaded

    raise ValueError("source must be one of: metadata_csv, local_dir, url_list")


@app.command()
def ingest(
    source: str = typer.Option("metadata_csv", "--source", help="Source type: metadata_csv | local_dir | url_list"),
    path: Path | None = typer.Option(None, "--path", help="Path to source (CSV file, directory, or URL list file)"),
    workers: int | None = typer.Option(None, "--workers", help="Parallel workers for PDF processing"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index", help="Force rebuild FAISS + BM25 index after ingest"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Re-process only previously failed documents"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed, do nothing"),
) -> None:
    settings = get_settings()
    root = find_project_root()
    chunks_path, embeddings_path = _canonical_paths(root)
    effective_workers = workers if workers is not None else settings.ingest_workers
    chunk_strategy = _chunk_strategy()
    pdf_dir = root / "data" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_impl = create_source(source, path)
        documents = source_impl.discover()
    except Exception as exc:
        logger.error("ingest_discovery_failed", error=str(exc))
        raise typer.Exit(code=2)

    if dry_run:
        typer.echo(
            json.dumps(
                {
                    "event": "ingest_dry_run",
                    "source": source,
                    "path": str(path) if path else "",
                    "workers": effective_workers,
                    "retry_failed": retry_failed,
                    "rebuild_index": rebuild_index,
                    "discovered": len(documents),
                    "files": [str(doc.candidate_path(pdf_dir)) for doc in documents],
                },
                ensure_ascii=False,
            )
        )
        return

    manifest = Manifest(root / "data" / "manifest.json")
    existing_chunks = _load_existing_chunks(root, chunks_path)
    chunks_by_doc = _group_chunks_by_doc(existing_chunks)
    seeded_count = _seed_manifest_from_existing_corpus(
        manifest,
        documents,
        chunks_by_doc,
        pdf_dir,
        chunk_strategy=chunk_strategy,
        embedding_model=settings.embedding_model,
    )
    if seeded_count:
        logger.info("manifest_bootstrapped", count=seeded_count, path=str(manifest.path))

    failed_files: list[str] = []
    processed_doc_ids: set[str] = set()
    skipped_docs = 0

    planned_jobs: list[tuple[DocumentMeta, Path, Any]] = []
    for doc in documents:
        try:
            pdf_path = source_impl.fetch(doc, pdf_dir)
            decision = manifest.should_process(
                doc.doc_id,
                pdf_path,
                chunk_strategy=chunk_strategy,
                embedding_model=settings.embedding_model,
                retry_failed_only=retry_failed,
            )
        except Exception as exc:
            failed_files.append(doc.doc_id)
            logger.warning("ingest_fetch_failed", doc_id=doc.doc_id, error=str(exc))
            continue

        if not decision.should_process:
            skipped_docs += 1
            logger.info("ingest_skipped", doc_id=doc.doc_id, action=decision.action)
            continue

        planned_jobs.append((doc, pdf_path, decision))

    chunk_results, chunk_errors = _extract_chunks_for_paths(
        [pdf_path for _, pdf_path, _ in planned_jobs],
        effective_workers,
    )
    for doc, pdf_path, decision in planned_jobs:
        error = chunk_errors.get(str(pdf_path))
        if error is not None:
            failed_files.append(str(pdf_path))
            manifest.set_failed(
                doc.doc_id,
                content_hash=decision.content_hash,
                file_size_bytes=decision.file_size_bytes,
                chunk_strategy=chunk_strategy,
                embedding_model=settings.embedding_model,
                error_message=error,
            )
            logger.warning("ingest_pdf_failed", file=str(pdf_path), error=error)
            continue

        chunks = chunk_results.get(str(pdf_path), [])
        chunks_by_doc[doc.doc_id] = chunks
        processed_doc_ids.add(doc.doc_id)
        manifest.set_complete(
            doc.doc_id,
            content_hash=decision.content_hash,
            file_size_bytes=decision.file_size_bytes,
            chunk_strategy=chunk_strategy,
            num_chunks=len(chunks),
            embedding_model=settings.embedding_model,
        )

    all_chunks = [chunk for doc_id in sorted(chunks_by_doc) for chunk in chunks_by_doc[doc_id]]

    if not all_chunks:
        logger.error("ingest_no_chunks")
        raise typer.Exit(code=2)

    generated_new_corpus = bool(processed_doc_ids) or not chunks_path.exists()
    embeddings_need_rebuild = generated_new_corpus or rebuild_index or not embeddings_path.exists()
    current_embedding_model = _embedding_file_model(embeddings_path)
    if not embeddings_need_rebuild and current_embedding_model != settings.embedding_model:
        embeddings_need_rebuild = True

    if generated_new_corpus or not chunks_path.exists():
        write_chunks_jsonl(all_chunks, chunks_path)

    embedded_records: list[EmbeddingRecord] = []
    if embeddings_need_rebuild:
        records_for_embedding = _records_for_embedding(all_chunks)
        vectors = embed_chunks(records_for_embedding, settings.embedding_model, settings.embedding_batch_size)
        now = _now_utc()
        for record, vector in zip(records_for_embedding, vectors):
            text = record["text"]
            embedded_records.append(
                EmbeddingRecord(
                    id=record["id"],
                    text=text,
                    metadata=record["metadata"],
                    embedding=vector,
                    content_hash=hashlib.md5(text.encode("utf-8")).hexdigest(),
                    embedding_model=settings.embedding_model,
                    created_at=now,
                )
            )

        write_embeddings_jsonl(embedded_records, embeddings_path)
        for doc_id, entry in list(manifest.entries.items()):
            if entry.status != "complete":
                continue
            manifest.set_complete(
                doc_id,
                content_hash=entry.content_hash,
                file_size_bytes=entry.file_size_bytes,
                chunk_strategy=entry.chunk_strategy,
                num_chunks=entry.num_chunks,
                embedding_model=settings.embedding_model,
                parsed_at=entry.parsed_at,
                embedded_at=now,
            )

    manifest.save()

    should_rebuild_indices = rebuild_index or generated_new_corpus or embeddings_need_rebuild
    faiss_store = FaissStore(embeddings_path=embeddings_path, embedding_model=settings.embedding_model)
    bm25_index = BM25Index(chunks_path=chunks_path)
    if not should_rebuild_indices:
        try:
            should_rebuild_indices = not (faiss_store.load() and bm25_index.load())
        except RuntimeError:
            should_rebuild_indices = True

    try:
        if should_rebuild_indices:
            index_info = _build_search_indices(chunks_path, embeddings_path, settings.embedding_model)
        else:
            index_info = {
                "faiss_index": str(faiss_store.index_path),
                "faiss_metadata": str(faiss_store.metadata_path),
                "text_data": str(faiss_store.text_data_path),
                "vectors": len(faiss_store.text_data),
                "bm25_index": str(bm25_index.index_path),
                "bm25_metadata": str(bm25_index.metadata_path),
                "bm25_rows": len(bm25_index.records),
            }
    except Exception as exc:
        logger.error("index_build_failed", error=str(exc))
        raise typer.Exit(code=2)

    status_code = 0 if not failed_files else 1
    typer.echo(
        json.dumps(
            {
                "event": "ingest_summary",
                "status": "ok" if status_code == 0 else "partial",
                "source": source,
                "processed_pdfs": len(processed_doc_ids),
                "failed_pdfs": len(failed_files),
                "skipped_pdfs": skipped_docs,
                "chunks_written": len(all_chunks),
                "embeddings_written": len(embedded_records) if embeddings_need_rebuild else _count_jsonl_rows(embeddings_path),
                "faiss_index": index_info["faiss_index"],
                "faiss_metadata": index_info["faiss_metadata"],
                "text_data": index_info["text_data"],
                "vectors": index_info["vectors"],
                "bm25_index": index_info["bm25_index"],
                "bm25_metadata": index_info["bm25_metadata"],
                "bm25_rows": index_info["bm25_rows"],
                "workers": effective_workers,
                "rebuild_index": rebuild_index,
                "indices_rebuilt": should_rebuild_indices,
            },
            ensure_ascii=False,
        )
    )
    if status_code != 0:
        raise typer.Exit(code=status_code)


@app.command()
def query(
    question: str = typer.Argument(..., metavar="QUESTION", help="The question string"),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve"),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="dense | sparse | hybrid"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable cross-encoder reranking"),
    llm: str | None = typer.Option(None, "--llm", help="LLM backend: gemini | ollama"),
    output: Path | None = typer.Option(None, "--output", help="Write JSON answer to file instead of stdout"),
) -> None:
    settings = get_settings()
    try:
        pipeline = RAGPipeline(
            config=RAGConfig(
                top_k=top_k,
                retrieval_mode=retrieval_mode or settings.retrieval_mode,
                rerank=rerank,
                llm_backend=llm or settings.llm_backend,
            )
        )
        answer = pipeline.answer_question(
            question,
            top_k=top_k,
            retrieval_mode=retrieval_mode or settings.retrieval_mode,
            rerank=rerank,
            llm_backend=llm or settings.llm_backend,
        )
    except Exception as exc:
        logger.error("query_generation_failed", error=str(exc))
        raise typer.Exit(code=1)

    payload = json.dumps(answer.model_dump(), ensure_ascii=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    else:
        typer.echo(payload)


@app.command()
def search(
    question: str = typer.Argument(..., metavar="QUESTION", help="The query string"),
    top_k: int = typer.Option(10, "--top-k", help="Number of results"),
    retrieval_mode: str | None = typer.Option(None, "--retrieval-mode", help="dense | sparse | hybrid"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Enable reranking"),
    output_format: str = typer.Option("json", "--output-format", help="json | table"),
) -> None:
    settings = get_settings()
    effective_mode = retrieval_mode or settings.retrieval_mode

    try:
        start = time.perf_counter()
        hits = _retrieve_results(question, top_k, effective_mode, rerank=rerank)
        retrieval_latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        logger.error("search_failed", error=str(exc))
        raise typer.Exit(code=1)

    results = []
    for hit in hits:
        results.append(
            {
                "rank": int(hit.get("rank", len(results) + 1)),
                "ref_id": hit.get("ref_id", ""),
                "score": float(hit.get("score", 0.0)),
                "text": hit.get("text", ""),
                "page": hit.get("page"),
                "source_file": hit.get("source_file", ""),
                "headings": hit.get("headings", []),
            }
        )

    if output_format == "table":
        typer.echo("rank\tref_id\tscore\tpage\tsource_file\ttext")
        for item in results:
            typer.echo(
                f"{item['rank']}\t{item['ref_id']}\t{item['score']:.6f}\t"
                f"{item['page']}\t{item['source_file']}\t{str(item['text'])[:120]}"
            )
    else:
        typer.echo(
            json.dumps(
                {
                    "results": results,
                    "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                    "retrieval_mode": effective_mode,
                    "total_results": len(results),
                },
                ensure_ascii=False,
            )
        )


@app.command()
def benchmark(
    dataset: Path = typer.Option(Path("data/train_QA.csv"), "--dataset", help="Path to labeled Q&A CSV"),
    retrieval_mode: str = typer.Option("hybrid", "--retrieval-mode", help="dense | sparse | hybrid"),
    top_k: int = typer.Option(5, "--top-k"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    output_dir: Path = typer.Option(Path("data/benchmark_results/"), "--output-dir", help="Directory for report output"),
    tag: str = typer.Option("run", "--tag", help="Human-readable tag for this run"),
) -> None:
    evaluator = Evaluator(
        dataset=dataset,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        rerank=rerank,
        output_dir=output_dir,
        tag=tag,
    )
    report_path, summary_path = evaluator.run()
    typer.echo(json.dumps({"report": str(report_path), "summary": str(summary_path)}, ensure_ascii=False))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    api = FastAPI(title="RAG API")

    @api.get("/api/v1/health")
    def health() -> dict:
        settings = get_settings()
        return {
            "status": "ok",
            "index_loaded": (find_project_root() / "data" / "my_faiss.index").exists(),
            "num_papers": 0,
            "num_chunks": 0,
            "embedding_model": settings.embedding_model,
            "retrieval_modes_available": ["dense", "sparse", "hybrid"],
        }

    uvicorn.run(api, host=host, port=port, reload=reload)
