"""Contextual re-ranking that blends semantic similarity with shopper context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import Product, UserPreferences
from .retrieval import RetrievedProduct


@dataclass
class RankedProduct:
    product: Product
    similarity: float
    context_score: float
    final_score: float


class ContextualRanker:
    """Re-ranks retrieved products using preference, price, and review-quality signals.

    ``final_score = similarity_weight * similarity + context_weight * context_score``

    ``context_score`` blends three signals that pure text similarity cannot see:
      - whether the product's category matches the shopper's stated preference
      - how comfortably the price fits the shopper's budget
      - the product's average review rating, as a quality/trust signal
    """

    def __init__(self, similarity_weight: float = 0.6, context_weight: float = 0.4) -> None:
        self._w_sim = similarity_weight
        self._w_ctx = context_weight

    def rank(
        self,
        candidates: List[RetrievedProduct],
        preferences: Optional[UserPreferences] = None,
    ) -> List[RankedProduct]:
        ranked = [self._score(candidate, preferences) for candidate in candidates]
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked

    def _score(
        self, candidate: RetrievedProduct, preferences: Optional[UserPreferences]
    ) -> RankedProduct:
        product = candidate.product
        context_score = self._context_score(product, preferences)
        final = self._w_sim * candidate.similarity + self._w_ctx * context_score
        return RankedProduct(
            product=product,
            similarity=candidate.similarity,
            context_score=context_score,
            final_score=final,
        )

    @staticmethod
    def _context_score(product: Product, preferences: Optional[UserPreferences]) -> float:
        rating_score = (product.average_rating or 3.0) / 5.0

        if preferences is None:
            return rating_score

        category_score = (
            1.0
            if (preferences.category is None or preferences.category == product.category)
            else 0.3
        )

        if preferences.max_price is None:
            price_score = 1.0
        elif product.price <= preferences.max_price:
            # Reward comfortably staying under budget without over-rewarding very cheap items.
            headroom = (preferences.max_price - product.price) / preferences.max_price
            price_score = 0.7 + 0.3 * min(headroom, 1.0)
        else:
            overage = (product.price - preferences.max_price) / preferences.max_price
            price_score = max(0.0, 0.5 - overage)

        return 0.5 * rating_score + 0.3 * category_score + 0.2 * price_score
