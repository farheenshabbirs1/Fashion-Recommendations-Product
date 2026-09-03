"""CLI demo: run one query through the full RAG recommendation pipeline.

Examples
--------
    python scripts/run_demo.py "something breathable for a beach vacation"
    python scripts/run_demo.py "office attire" --category bottoms --max-price 100
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... python scripts/run_demo.py "date night look"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_rag.data_loader import load_products  # noqa: E402
from fashion_rag.models import UserPreferences  # noqa: E402
from fashion_rag.pipeline import RecommendationPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fashion product recommendation query.")
    parser.add_argument(
        "query",
        nargs="?",
        default="something breathable and casual for a beach vacation",
        help="Free-text shopper request.",
    )
    parser.add_argument("--category", default=None, help="Preferred product category.")
    parser.add_argument("--max-price", type=float, default=None, help="Budget ceiling in USD.")
    parser.add_argument("--embedder", default=None, help="tfidf (default), sbert, or openai")
    parser.add_argument("--llm", default=None, help="mock (default), openai, or anthropic")
    args = parser.parse_args()

    products = load_products()
    pipeline = RecommendationPipeline.build(
        products, embedder_name=args.embedder, llm_provider=args.llm
    )

    preferences = UserPreferences(category=args.category, max_price=args.max_price)
    result = pipeline.recommend(args.query, preferences=preferences)

    print(f"Query: {result.query}\n")
    print("Top ranked candidates:")
    for rp in result.ranked_products[:5]:
        print(
            f"  {rp.product.name:30s}  sim={rp.similarity:.3f}  "
            f"ctx={rp.context_score:.3f}  final={rp.final_score:.3f}"
        )
    print("\nGenerated recommendation:\n")
    print(result.generated_text)


if __name__ == "__main__":
    main()
