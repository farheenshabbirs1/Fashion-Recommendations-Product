"""Retrieval-quality evaluation: precision / recall / F1 against labeled queries.

Compares the semantic (embedding-based) retriever against a naive keyword
baseline to quantify the quality gain from embedding-based retrieval, the
number ``scripts/run_evaluation.py`` reports at the end of its run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .models import Product, UserPreferences


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float


def precision_recall_f1(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> EvalResult:
    retrieved_set, relevant_set = set(retrieved_ids), set(relevant_ids)
    if not retrieved_set:
        return EvalResult(0.0, 0.0, 0.0)
    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set)
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EvalResult(precision, recall, f1)


def keyword_baseline(products: List[Product], query: str, top_k: int) -> List[str]:
    """Naive baseline: rank products by raw word-overlap count with the query."""
    query_terms = set(query.lower().split())
    scored = []
    for product in products:
        doc_terms = set(product.to_document().lower().split())
        overlap = len(query_terms & doc_terms)
        scored.append((overlap, product.id))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [product_id for _, product_id in scored[:top_k]]


def evaluate_system(
    recommend_fn: Callable[[str, UserPreferences, int], List[str]],
    products: List[Product],
    eval_queries: List[dict],
    top_k: int = 5,
) -> Dict[str, EvalResult]:
    """Average F1 for the semantic system vs. the keyword baseline across eval_queries."""
    system_scores, baseline_scores = [], []
    for item in eval_queries:
        query = item["query"]
        relevant_ids = item["relevant_product_ids"]
        preferences = UserPreferences(**item.get("preferences", {}))

        system_ids = recommend_fn(query, preferences, top_k)
        system_scores.append(precision_recall_f1(system_ids, relevant_ids))

        baseline_ids = keyword_baseline(products, query, top_k)
        baseline_scores.append(precision_recall_f1(baseline_ids, relevant_ids))

    return {
        "system": _average(system_scores),
        "baseline": _average(baseline_scores),
    }


def _average(results: List[EvalResult]) -> EvalResult:
    n = len(results) or 1
    return EvalResult(
        precision=sum(r.precision for r in results) / n,
        recall=sum(r.recall for r in results) / n,
        f1=sum(r.f1 for r in results) / n,
    )
