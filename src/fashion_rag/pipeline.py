"""End-to-end RAG recommendation pipeline: retrieve -> rank -> prompt -> generate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .llm_client import LLMClient, get_llm_client
from .models import Product, UserPreferences
from .prompting import PromptBuilder
from .ranking import ContextualRanker, RankedProduct
from .retrieval import SemanticRetriever


@dataclass
class RecommendationResult:
    query: str
    ranked_products: List[RankedProduct]
    prompt: str
    generated_text: str


class RecommendationPipeline:
    """Wires together retrieval, ranking, prompting, and generation."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        ranker: Optional[ContextualRanker] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self._retriever = retriever
        self._ranker = ranker or ContextualRanker()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._llm_client = llm_client or get_llm_client()

    @classmethod
    def build(
        cls,
        products: Sequence[Product],
        embedder_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> "RecommendationPipeline":
        """Convenience constructor: builds the retriever's index from a product catalog."""
        from .embeddings import get_embedder

        retriever = SemanticRetriever(embedder=get_embedder(embedder_name))
        retriever.index(products)
        return cls(retriever=retriever, llm_client=get_llm_client(llm_provider))

    def recommend(
        self,
        query: str,
        preferences: Optional[UserPreferences] = None,
        top_k: int = 8,
        top_n_in_prompt: int = 3,
    ) -> RecommendationResult:
        candidates = self._retriever.search(query, top_k=top_k)
        ranked = self._ranker.rank(candidates, preferences)
        prompt = self._prompt_builder.build(query, ranked, preferences, top_n=top_n_in_prompt)
        generated = self._llm_client.generate(prompt)
        return RecommendationResult(
            query=query, ranked_products=ranked, prompt=prompt, generated_text=generated
        )
