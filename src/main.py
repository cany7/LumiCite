from __future__ import annotations

import csv
import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import faiss
import numpy as np
import pandas as pd
import requests
import typer
import uvicorn
from fastapi import FastAPI

from src.config.settings import get_settings
from src.core.constants import FALLBACK_ANSWER
from src.core.logging import get_logger
from src.core.paths import find_project_root
from src.core.schemas import Citation, EmbeddingRecord, RAGAnswer, VerificationResult
from src.generation.generator import LLMGenerator
from src.generation.prompt_templates import build_prompt
from src.indexing.ollama_generator import rag_ollama_answer
from src.indexing.retrieval import get_chunks
from src.ingestion.chunker import extract_pdf_chunks, write_chunks_jsonl
from src.ingestion.embedder import embed_chunks, load_canonical_chunks_jsonl, write_embeddings_jsonl
from src.ingestion.legacy_loader import load_legacy_chunks_json

app = typer.Typer(help="RAG command line interface")
logger = get_logger(__name__)


def _now_utc() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _canonical_paths(root: Path) -> tuple[Path, Path]:
    json_dir = root / "data" / "JSON"
    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir / "chunks.jsonl", json_dir / "embeddings.jsonl"


def _build_faiss_index(embeddings_path: Path) -> tuple[Path, Path, int]:
    root = find_project_root()
    index_path = root / "data" / "my_faiss.index"
    text_data_path = root / "data" / "text_data.pkl"

    embeddings: list[list[float]] = []
    text_data: list[dict] = []
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            embeddings.append(record["embedding"])
            text_data.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record.get("metadata", {}),
                }
            )

    if not embeddings:
        raise ValueError("No embeddings to index")

    vectors = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with text_data_path.open("wb") as handle:
        pickle.dump(text_data, handle)

    return index_path, text_data_path, int(index.ntotal)


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

    try:
        pdf_files = _discover_pdfs(source, path)
    except Exception as exc:
        logger.error("ingest_discovery_failed", error=str(exc))
        raise typer.Exit(code=2)

    if dry_run:
        typer.echo(
            json.dumps(
                {
                    "source": source,
                    "path": str(path) if path else "",
                    "workers": effective_workers,
                    "retry_failed": retry_failed,
                    "rebuild_index": rebuild_index,
                    "discovered": len(pdf_files),
                    "files": [str(p) for p in pdf_files],
                },
                ensure_ascii=False,
            )
        )
        return

    all_chunks = []
    failed_files: list[str] = []

    for pdf_path in pdf_files:
        try:
            chunks = extract_pdf_chunks(pdf_path)
            all_chunks.extend(chunks)
        except Exception as exc:
            failed_files.append(str(pdf_path))
            logger.warning("ingest_pdf_failed", file=str(pdf_path), error=str(exc))

    if not all_chunks:
        if chunks_path.exists() and not rebuild_index:
            all_chunks = load_canonical_chunks_jsonl(chunks_path)
        else:
            legacy_sources = [root / "data" / "JSON" / "chunks.json", root / "data" / "JSON" / "alt_text.json"]
            for legacy in legacy_sources:
                if legacy.exists():
                    all_chunks.extend(load_legacy_chunks_json(legacy))

    if not all_chunks:
        logger.error("ingest_no_chunks")
        raise typer.Exit(code=2)

    write_chunks_jsonl(all_chunks, chunks_path)

    records_for_embedding: list[dict] = []
    for chunk in all_chunks:
        records_for_embedding.append(
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
        )

    vectors = embed_chunks(records_for_embedding, settings.embedding_model, settings.embedding_batch_size)
    now = _now_utc()
    embedded_records: list[EmbeddingRecord] = []
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

    try:
        index_path, text_data_path, total_vectors = _build_faiss_index(embeddings_path)
    except Exception as exc:
        logger.error("index_build_failed", error=str(exc))
        raise typer.Exit(code=2)

    status_code = 0 if not failed_files else 1
    typer.echo(
        json.dumps(
            {
                "status": "ok" if status_code == 0 else "partial",
                "source": source,
                "processed_pdfs": len(pdf_files) - len(failed_files),
                "failed_pdfs": len(failed_files),
                "chunks_written": len(all_chunks),
                "embeddings_written": len(embedded_records),
                "faiss_index": str(index_path),
                "text_data": str(text_data_path),
                "vectors": total_vectors,
                "workers": effective_workers,
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
    effective_mode = retrieval_mode or settings.retrieval_mode
    effective_llm = llm or settings.llm_backend

    try:
        retrieval_start = time.perf_counter()
        hits = get_chunks(question, num_chunks=top_k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
    except Exception as exc:
        logger.error("query_retrieval_failed", error=str(exc))
        raise typer.Exit(code=1)

    citations: list[Citation] = []
    contexts = []
    candidate_ref_ids = []
    for rank in sorted(hits.keys()):
        hit = hits[rank]
        ref_id = str(hit.get("paper", ""))
        candidate_ref_ids.append(ref_id)
        contexts.append(
            {
                "ref_id": ref_id,
                "text": hit.get("chunk", ""),
                "page": hit.get("page"),
                "headings": hit.get("headings", []),
            }
        )
        citations.append(
            Citation(
                ref_id=ref_id,
                page=hit.get("page"),
                evidence_text=str(hit.get("chunk", ""))[:300],
                evidence_type="text",
            )
        )

    generation_start = time.perf_counter()
    try:
        if contexts:
            if effective_llm == "ollama":
                raw = rag_ollama_answer(
                    question,
                    hits,
                    model=None,
                    ollama_url=settings.ollama_base_url,
                )
            else:
                prompt = build_prompt(question, contexts, candidate_ref_ids)
                llm_client = LLMGenerator(
                    model=settings.gemini_model,
                    project=settings.gcp_project,
                    location=settings.gcp_location,
                    credentials_path=settings.gcp_credentials_path,
                )
                raw = llm_client.generate_json(prompt)
        else:
            raw = {
                "answer": FALLBACK_ANSWER,
                "answer_value": "is_blank",
                "answer_unit": "is_blank",
                "ref_id": [],
                "supporting_materials": "is_blank",
                "explanation": "is_blank",
            }
    except Exception as exc:
        logger.error("query_generation_failed", error=str(exc))
        raise typer.Exit(code=1)

    generation_latency_ms = (time.perf_counter() - generation_start) * 1000

    answer = RAGAnswer(
        answer=str(raw.get("answer") or FALLBACK_ANSWER),
        answer_value=str(raw.get("answer_value") or "is_blank"),
        answer_unit=str(raw.get("answer_unit") or "is_blank"),
        ref_id=[str(r) for r in (raw.get("ref_id") or [])],
        supporting_materials=str(raw.get("supporting_materials") or "is_blank"),
        explanation=str(raw.get("explanation") or "is_blank"),
        citations=citations,
        retrieval_latency_ms=round(retrieval_latency_ms, 3),
        generation_latency_ms=round(generation_latency_ms, 3),
        retrieval_mode=effective_mode,
        llm_backend=effective_llm,
        verification=VerificationResult(
            passed=bool(contexts) and str(raw.get("answer") or "") != FALLBACK_ANSWER,
            confidence=0.7 if contexts else 0.0,
            warnings=[] if contexts else ["No retrieved context above threshold."],
            corrected_output=None,
        ),
    )

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
        hits = get_chunks(question, num_chunks=top_k)
        retrieval_latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        logger.error("search_failed", error=str(exc))
        raise typer.Exit(code=1)

    results = []
    for rank in sorted(hits.keys()):
        hit = hits[rank]
        results.append(
            {
                "rank": rank,
                "ref_id": hit.get("paper", ""),
                "score": float(hit.get("score", 0.0)),
                "text": hit.get("chunk", ""),
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

    if rerank:
        logger.info("rerank_requested_but_not_available_in_phase1", requested=True)


@app.command()
def benchmark(
    dataset: Path = typer.Option(Path("data/train_QA.csv"), "--dataset", help="Path to labeled Q&A CSV"),
    retrieval_mode: str = typer.Option("hybrid", "--retrieval-mode", help="dense | sparse | hybrid"),
    top_k: int = typer.Option(5, "--top-k"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    output_dir: Path = typer.Option(Path("data/benchmark_results/"), "--output-dir", help="Directory for report output"),
    tag: str = typer.Option("run", "--tag", help="Human-readable tag for this run"),
) -> None:
    settings = get_settings()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"{tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_questions = 0
    latencies = []
    if dataset.exists():
        df = pd.read_csv(dataset)
        num_questions = len(df)
        if "question" in df.columns:
            for q in df["question"].head(min(20, num_questions)).tolist():
                start = time.perf_counter()
                try:
                    get_chunks(str(q), num_chunks=top_k)
                    latencies.append((time.perf_counter() - start) * 1000)
                except Exception:
                    latencies.append((time.perf_counter() - start) * 1000)

    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    report = {
        "run_id": run_id,
        "tag": tag,
        "timestamp": _now_utc(),
        "config_hash": hashlib.md5(f"{retrieval_mode}:{top_k}:{rerank}".encode("utf-8")).hexdigest(),
        "git_commit": "",
        "dataset": str(dataset),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "reranker_enabled": rerank,
        "embedding_model": settings.embedding_model,
        "num_questions": num_questions,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
        "mean_retrieval_latency_ms": round(mean_latency, 3),
        "p95_retrieval_latency_ms": round(p95_latency, 3),
        "per_question": [],
    }

    report_path = output_dir / f"{run_id}_report.json"
    summary_path = output_dir / f"{run_id}_summary.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "retrieval_mode", "top_k", "rerank", "mean_retrieval_latency_ms", "p95_retrieval_latency_ms"])
        writer.writerow([run_id, retrieval_mode, top_k, int(rerank), round(mean_latency, 3), round(p95_latency, 3)])

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
