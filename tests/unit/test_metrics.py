from __future__ import annotations

import pytest

from src.evaluation.metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_perfect():
    assert recall_at_k(["paper1", "paper2"], ["paper2"], k=5) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["paper1", "paper2"], ["paper3"], k=5) == 0.0


def test_mrr_first_rank():
    assert mrr(["paper1", "paper2"], ["paper1"]) == 1.0


def test_mrr_third_rank():
    assert mrr(["paper1", "paper2", "paper3"], ["paper3"]) == pytest.approx(1 / 3)


def test_mrr_miss():
    assert mrr(["paper1", "paper2"], ["paper3"]) == 0.0


def test_ndcg_at_k_perfect():
    assert ndcg_at_k(["paper1", "paper2"], ["paper1", "paper2"], k=2) == 1.0


def test_ndcg_at_k_empty():
    assert ndcg_at_k(["paper1"], [], k=5) == 0.0
