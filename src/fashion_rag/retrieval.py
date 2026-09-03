"""Semantic (embedding-based) retrieval over the product catalog."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .embeddings import Embedder
from .models import Product


@dataclass
class RetrievedProduct:
    product: Product
    similarity: float


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norm @ query_norm


def _chunk(items: List[str], n: int) -> List[List[str]]:
    if not items:
        return [[]]
    n = max(1, min(n, len(items)))
    size = -(-len(items) // n)  # ceil division
    return [items[i : i + size] for i in range(0, len(items), size)]


class SemanticRetriever:
    """Builds an embedding index over products (+ their reviews) and retrieves top-k matches."""

    def __init__(self, embedder: Embedder, max_workers: int = 8) -> None:
        self._embedder = embedder
        self._max_workers = max_workers
        self._products: List[Product] = []
        self._matrix: Optional[np.ndarray] = None

    def index(self, products: Sequence[Product]) -> None:
        """Embed every product's document text, in parallel batches, and build the index.

        Splitting the corpus into chunks and embedding them concurrently
        overlaps the per-call latency of API-backed embedders (OpenAI,
        sentence-transformers batches) across the catalog, cutting wall-clock
        indexing time roughly by the number of workers instead of paying for
        each document sequentially.
        """
        self._products = list(products)
        documents = [p.to_document() for p in self._products]
        self._embedder.fit(documents)

        chunks = _chunk(documents, self._max_workers)
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            embedded_chunks = list(pool.map(self._embedder.embed, chunks))
        non_empty = [c for c in embedded_chunks if len(c)]
        self._matrix = np.vstack(non_empty) if non_empty else np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedProduct]:
        if self._matrix is None:
            raise RuntimeError("Call index() before search().")
        query_vec = self._embedder.embed([query])[0]
        scores = _cosine_similarity(query_vec, self._matrix)
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedProduct(product=self._products[i], similarity=float(scores[i]))
            for i in top_indices
        ]

    def search_many(self, queries: Sequence[str], top_k: int = 5) -> List[List[RetrievedProduct]]:
        """Run several queries concurrently -- used by the batch evaluation script."""
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(lambda q: self.search(q, top_k), queries))
