from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_settings, normalize_reasoning_effort
from src.core.schemas import BenchmarkReport
from src.evaluation.metrics import mrr, ndcg_at_k, recall_at_k
from src.retrieval import get_retriever
from src.retrieval.query_explanation import QueryExplanationConfig, retrieve_with_optional_query_explanation
from src.retrieval.reranker import Reranker


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_ref_doc_ids(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text == "is_blank":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return ""


class Evaluator:
    def __init__(
        self,
        dataset: Path,
        retrieval_mode: str,
        top_k: int,
        rerank: bool,
        query_explanation: bool,
        output_dir: Path,
        tag: str,
    ) -> None:
        self.dataset = dataset
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.rerank = rerank
        self.query_explanation = query_explanation
        self.output_dir = output_dir
        self.tag = tag
        self.settings = get_settings()
        self.retriever = get_retriever(self.retrieval_mode)
        self.reranker = Reranker() if self.rerank else None

    def _retrieve(self, question: str) -> tuple[list[dict], float]:
        start = time.perf_counter()
        if self.query_explanation:
            execution = retrieve_with_optional_query_explanation(
                question,
                top_k=self.top_k,
                retrieval_mode=self.retrieval_mode,
                rerank=self.rerank,
                query_explanation=QueryExplanationConfig(
                    enabled=True,
                    llm_model=self.settings.api_model,
                    api_key=self.settings.api_key,
                    base_url=self.settings.api_base_url,
                    reasoning_effort=normalize_reasoning_effort(self.settings.query_explanation_reasoning_effort),
                ),
            )
            results = execution.results
        else:
            fetch_k = self.top_k * 5 if self.rerank else self.top_k
            results = self.retriever.retrieve(question, fetch_k)
            if self.reranker is not None:
                results = self.reranker.rerank(question, results, self.top_k)
            else:
                results = results[: self.top_k]
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms

    def run(self) -> tuple[Path, Path]:
        frame = pd.read_csv(self.dataset)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.tag}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        per_question: list[dict] = []
        recall_scores: list[float] = []
        mrr_scores: list[float] = []
        ndcg_scores: list[float] = []
        latencies: list[float] = []

        for row in frame.to_dict(orient="records"):
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            question_id = str(row.get("question_id", row.get("id", ""))).strip()
            relevant_ref_doc_ids = _parse_ref_doc_ids(row.get("ref_doc_id", row.get("ref_id")))
            results, latency_ms = self._retrieve(question)
            retrieved_doc_ids = [str(item.get("doc_id", "")).strip() for item in results if item.get("doc_id")]

            recall_value = recall_at_k(retrieved_doc_ids, relevant_ref_doc_ids, self.top_k)
            mrr_value = mrr(retrieved_doc_ids, relevant_ref_doc_ids)
            ndcg_value = ndcg_at_k(retrieved_doc_ids, relevant_ref_doc_ids, self.top_k)

            recall_scores.append(recall_value)
            mrr_scores.append(mrr_value)
            ndcg_scores.append(ndcg_value)
            latencies.append(latency_ms)
            per_question.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "relevant_ref_doc_ids": relevant_ref_doc_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "retrieved_ref_ids": retrieved_doc_ids,
                    "recall_at_k": recall_value,
                    "mrr": mrr_value,
                    "ndcg_at_k": ndcg_value,
                    "retrieval_latency_ms": round(latency_ms, 3),
                }
            )

        report = BenchmarkReport(
            run_id=run_id,
            tag=self.tag,
            timestamp=_now_utc(),
            config_hash=hashlib.md5(
                f"{self.retrieval_mode}:{self.top_k}:{self.rerank}:{self.query_explanation}".encode("utf-8")
            ).hexdigest(),
            git_commit=_git_commit(),
            dataset=str(self.dataset),
            retrieval_mode=self.retrieval_mode,
            top_k=self.top_k,
            reranker_enabled=self.rerank,
            embedding_model=self.settings.embedding_model,
            num_questions=len(per_question),
            recall_at_k=float(np.mean(recall_scores)) if recall_scores else 0.0,
            mrr=float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            ndcg_at_k=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            mean_retrieval_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            p95_retrieval_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            per_question=per_question,
        )

        report_path = self.output_dir / f"{run_id}_report.json"
        summary_path = self.output_dir / f"{run_id}_summary.csv"
        report_path.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "run_id",
                    "retrieval_mode",
                    "top_k",
                    "rerank",
                    "recall_at_k",
                    "mrr",
                    "ndcg_at_k",
                    "mean_retrieval_latency_ms",
                    "p95_retrieval_latency_ms",
                ]
            )
            writer.writerow(
                [
                    report.run_id,
                    report.retrieval_mode,
                    report.top_k,
                    int(report.reranker_enabled),
                    report.recall_at_k,
                    report.mrr,
                    report.ndcg_at_k,
                    report.mean_retrieval_latency_ms,
                    report.p95_retrieval_latency_ms,
                ]
            )
        return report_path, summary_path
