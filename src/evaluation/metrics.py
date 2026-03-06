from __future__ import annotations

import math


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def recall_at_k(retrieved_ref_ids: list[str], relevant_ref_ids: list[str], k: int) -> float:
    relevant = set(_unique_in_order(relevant_ref_ids))
    if k <= 0 or not relevant:
        return 0.0

    retrieved = set(_unique_in_order(retrieved_ref_ids)[:k])
    return len(retrieved & relevant) / len(relevant)


def mrr(retrieved_ref_ids: list[str], relevant_ref_ids: list[str]) -> float:
    relevant = set(_unique_in_order(relevant_ref_ids))
    if not relevant:
        return 0.0

    for rank, ref_id in enumerate(_unique_in_order(retrieved_ref_ids), start=1):
        if ref_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ref_ids: list[str], relevant_ref_ids: list[str], k: int) -> float:
    relevant = set(_unique_in_order(relevant_ref_ids))
    if k <= 0 or not relevant:
        return 0.0

    ranked = _unique_in_order(retrieved_ref_ids)[:k]
    dcg = 0.0
    for rank, ref_id in enumerate(ranked, start=1):
        if ref_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
