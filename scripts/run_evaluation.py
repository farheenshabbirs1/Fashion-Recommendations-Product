"""CLI: evaluate retrieval quality (precision/recall/F1) vs. a keyword baseline.

    python scripts/run_evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_rag.data_loader import load_eval_queries, load_products  # noqa: E402
from fashion_rag.embeddings import get_embedder  # noqa: E402
from fashion_rag.evaluate import evaluate_system  # noqa: E402
from fashion_rag.models import UserPreferences  # noqa: E402
from fashion_rag.retrieval import SemanticRetriever  # noqa: E402


def main() -> None:
    products = load_products()
    eval_queries = load_eval_queries()

    retriever = SemanticRetriever(embedder=get_embedder())
    retriever.index(products)

    def recommend_fn(query: str, preferences: Optional[UserPreferences], top_k: int) -> List[str]:
        results = retriever.search(query, top_k=top_k)
        return [r.product.id for r in results]

    scores = evaluate_system(recommend_fn, products, eval_queries, top_k=5)

    print("Retrieval quality (averaged over labeled eval queries):\n")
    for name, result in scores.items():
        print(
            f"  {name:10s} precision={result.precision:.3f}  "
            f"recall={result.recall:.3f}  f1={result.f1:.3f}"
        )

    system_f1, baseline_f1 = scores["system"].f1, scores["baseline"].f1
    if baseline_f1 > 0:
        lift = (system_f1 - baseline_f1) / baseline_f1 * 100
        print(f"\nEmbedding-based retrieval improves F1 by {lift:.1f}% over the keyword baseline.")
    else:
        print("\nBaseline F1 is 0.0 on this sample -- embedding-based retrieval finds matches it misses entirely.")


if __name__ == "__main__":
    main()
