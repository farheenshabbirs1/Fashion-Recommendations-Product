"""Embedding backends for semantic retrieval.

The pipeline is provider-agnostic: it defaults to a local TF-IDF embedder
that needs no network access or API keys, and can optionally switch to a
sentence-transformers model or an OpenAI embedding model when available.
"""
from __future__ import annotations

import abc
import os
from typing import List, Optional

import numpy as np


class Embedder(abc.ABC):
    """Base interface every embedding backend implements."""

    @abc.abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Fit the embedder on the full corpus (a no-op for stateless embedders)."""

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n, d) matrix of embeddings for the given texts."""


class TfidfEmbedder(Embedder):
    """Default, dependency-light embedder: TF-IDF + cosine similarity.

    Runs fully offline with no API keys or downloads. TF-IDF over unigrams
    and bigrams, with English stop words removed, is a strong and fully
    explainable baseline for small-to-medium product catalogs.
    """

    def __init__(self, max_features: int = 4096) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed().")
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Optional dense embedder backed by sentence-transformers, if installed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise ImportError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` or use the default TfidfEmbedder."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: List[str]) -> None:
        return  # pretrained dense embedders need no corpus-specific fitting

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """Optional embedder using OpenAI's embeddings API. Requires OPENAI_API_KEY."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise ImportError(
                "openai is not installed. Install it with `pip install openai`."
            ) from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def fit(self, corpus: List[str]) -> None:
        return  # no local state to fit; embeddings are computed per call

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model_name, input=texts)
        return np.asarray([d.embedding for d in response.data], dtype=np.float32)


def get_embedder(name: Optional[str] = None) -> Embedder:
    """Factory: choose an embedder by explicit name, the ``EMBEDDER`` env var, or a safe default.

    Resolution order: explicit ``name`` -> ``EMBEDDER`` env var -> ``"tfidf"``.
    """
    choice = (name or os.getenv("EMBEDDER") or "tfidf").lower()
    if choice == "tfidf":
        return TfidfEmbedder()
    if choice in {"sentence-transformers", "sbert"}:
        return SentenceTransformerEmbedder()
    if choice == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown embedder '{choice}'. Use 'tfidf', 'sbert', or 'openai'.")
