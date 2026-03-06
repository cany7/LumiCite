from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

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
from src.evaluation.evaluator import Evaluator
from src.generation.generator import LLMGenerator
from src.generation.prompt_templates import build_prompt
from src.indexing.bm25_index import BM25Index
from src.indexing.ollama_generator import rag_ollama_answer
from src.indexing.vector_store import FaissStore
from src.retrieval import get_retriever
from src.ingestion.chunker import extract_pdf_chunks, write_chunks_jsonl
from src.ingestion.embedder import embed_chunks, load_canonical_chunks_jsonl, write_embeddings_jsonl
from src.ingestion.legacy_loader import load_legacy_chunks_json
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
    generated_new_corpus = False

    for pdf_path in pdf_files:
        try:
            chunks = extract_pdf_chunks(pdf_path)
            all_chunks.extend(chunks)
            generated_new_corpus = True
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
                    generated_new_corpus = True

    if not all_chunks:
        logger.error("ingest_no_chunks")
        raise typer.Exit(code=2)

    embeddings_need_rebuild = generated_new_corpus or rebuild_index or not embeddings_path.exists()
    current_embedding_model = _embedding_file_model(embeddings_path)
    if not embeddings_need_rebuild and current_embedding_model != settings.embedding_model:
        embeddings_need_rebuild = True

    if generated_new_corpus or not chunks_path.exists():
        write_chunks_jsonl(all_chunks, chunks_path)

    embedded_records: list[EmbeddingRecord] = []
    if embeddings_need_rebuild:
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
                "status": "ok" if status_code == 0 else "partial",
                "source": source,
                "processed_pdfs": len(pdf_files) - len(failed_files),
                "failed_pdfs": len(failed_files),
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
    effective_mode = retrieval_mode or settings.retrieval_mode
    effective_llm = llm or settings.llm_backend

    try:
        retrieval_start = time.perf_counter()
        hits = _retrieve_results(question, top_k, effective_mode, rerank=rerank)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
    except Exception as exc:
        logger.error("query_retrieval_failed", error=str(exc))
        raise typer.Exit(code=1)

    citations: list[Citation] = []
    contexts = []
    candidate_ref_ids = []
    for hit in hits:
        ref_id = str(hit.get("ref_id", ""))
        candidate_ref_ids.append(ref_id)
        contexts.append(
            {
                "ref_id": ref_id,
                "text": hit.get("text", ""),
                "page": hit.get("page"),
                "headings": hit.get("headings", []),
            }
        )
        citations.append(
            Citation(
                ref_id=ref_id,
                page=hit.get("page"),
                evidence_text=str(hit.get("text", ""))[:300],
                evidence_type=str(hit.get("chunk_type", "text")),
            )
        )

    generation_start = time.perf_counter()
    try:
        if contexts:
            if effective_llm == "ollama":
                raw = rag_ollama_answer(
                    question,
                    {
                        item["rank"]: {
                            "chunk": item["text"],
                            "paper": item["ref_id"],
                            "rank": item["rank"],
                            "score": item["score"],
                            "page": item["page"],
                            "source_file": item["source_file"],
                            "headings": item["headings"],
                        }
                        for item in hits
                    },
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
