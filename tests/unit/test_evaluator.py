from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.evaluation.evaluator as evaluator_module
from src.evaluation.evaluator import Evaluator


class FakeRetriever:
    def __init__(self, results_by_question: dict[str, list[dict]]) -> None:
        self.results_by_question = results_by_question

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        return [dict(item) for item in self.results_by_question[question][:top_k]]


class FakeReranker:
    def __init__(self, reranked_by_question: dict[str, list[dict]]) -> None:
        self.reranked_by_question = reranked_by_question

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        return [dict(item) for item in self.reranked_by_question[query][:top_k]]


def _result(ref_id: str, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": f"{ref_id}_aaaabbbb",
        "ref_id": ref_id,
        "score": 1.0 / rank,
        "text": f"evidence for {ref_id}",
        "page": rank,
        "source_file": f"{ref_id}.pdf",
        "headings": [],
        "chunk_type": "text",
    }


def _write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "ref_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_evaluator_run_writes_report_and_summary_files(monkeypatch, tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    _write_dataset(
        dataset,
        [
            {"id": "q1", "question": "first question", "ref_id": "paper1"},
            {"id": "q2", "question": "second question", "ref_id": "paper3"},
        ],
    )
    fake_retriever = FakeRetriever(
        {
            "first question": [_result("paper1", 1), _result("paper2", 2)],
            "second question": [_result("paper2", 1), _result("paper3", 2)],
        }
    )

    monkeypatch.setattr(evaluator_module, "get_settings", lambda: SimpleNamespace(embedding_model="mock-model"))
    monkeypatch.setattr(evaluator_module, "get_retriever", lambda mode: fake_retriever)
    monkeypatch.setattr(evaluator_module, "_git_commit", lambda: "abc123")

    report_path, summary_path = Evaluator(
        dataset=dataset,
        retrieval_mode="hybrid",
        top_k=2,
        rerank=False,
        output_dir=tmp_path,
        tag="phase2",
    ).run()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path.exists()
    assert summary_path.exists()
    assert report["tag"] == "phase2"
    assert report["retrieval_mode"] == "hybrid"
    assert report["num_questions"] == 2
    assert report["embedding_model"] == "mock-model"
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == pytest.approx(0.75)
    assert report["ndcg_at_k"] == pytest.approx((1.0 + (1.0 / math.log2(3))) / 2)

    rows = list(csv.reader(summary_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0][:4] == ["run_id", "retrieval_mode", "top_k", "rerank"]
    assert rows[1][1:4] == ["hybrid", "2", "0"]


def test_evaluator_run_uses_reranked_results(monkeypatch, tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    _write_dataset(dataset, [{"id": "q1", "question": "rerank question", "ref_id": "paper1"}])
    fake_retriever = FakeRetriever({"rerank question": [_result("paper2", 1), _result("paper1", 2)]})
    fake_reranker = FakeReranker({"rerank question": [_result("paper1", 1)]})

    monkeypatch.setattr(evaluator_module, "get_settings", lambda: SimpleNamespace(embedding_model="mock-model"))
    monkeypatch.setattr(evaluator_module, "get_retriever", lambda mode: fake_retriever)
    monkeypatch.setattr(evaluator_module, "Reranker", lambda: fake_reranker)
    monkeypatch.setattr(evaluator_module, "_git_commit", lambda: "abc123")

    report_path, _summary_path = Evaluator(
        dataset=dataset,
        retrieval_mode="hybrid",
        top_k=1,
        rerank=True,
        output_dir=tmp_path,
        tag="phase2_rerank",
    ).run()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reranker_enabled"] is True
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["per_question"][0]["retrieved_ref_ids"] == ["paper1"]
