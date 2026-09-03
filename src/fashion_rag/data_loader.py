"""Loads the bundled synthetic product/review dataset from ``data/``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import Product, Review

# src/fashion_rag/data_loader.py -> parents[2] is the project root, next to data/
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_products(data_dir: Path = DATA_DIR) -> List[Product]:
    """Load products.json + reviews.json and join reviews onto their products."""
    products_raw = json.loads((data_dir / "products.json").read_text())
    reviews_raw = json.loads((data_dir / "reviews.json").read_text())

    reviews_by_product: Dict[str, List[Review]] = {}
    for r in reviews_raw:
        review = Review(product_id=r["product_id"], rating=r["rating"], text=r["text"])
        reviews_by_product.setdefault(review.product_id, []).append(review)

    products = []
    for p in products_raw:
        products.append(
            Product(
                id=p["id"],
                name=p["name"],
                category=p["category"],
                price=p["price"],
                tags=p["tags"],
                description=p["description"],
                reviews=reviews_by_product.get(p["id"], []),
            )
        )
    return products


def load_eval_queries(data_dir: Path = DATA_DIR) -> List[dict]:
    """Load the labeled evaluation queries used by ``scripts/run_evaluation.py``."""
    return json.loads((data_dir / "eval_queries.json").read_text())
