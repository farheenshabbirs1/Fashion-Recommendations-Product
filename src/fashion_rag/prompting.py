"""Prompt construction with basic token-budget-aware optimization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import Product, UserPreferences
from .ranking import RankedProduct

# Crude proxy for a token budget: keeps prompts small, predictable, and cheap
# to run regardless of how large the catalog or the reviews are.
MAX_CONTEXT_CHARS = 2400


@dataclass
class PromptBuilder:
    system_prompt: str = (
        "You are a helpful, concise fashion shopping assistant. Recommend "
        "products strictly from the provided context, explain briefly why "
        "each one fits the customer's request, and never invent products, "
        "prices, or reviews that are not in the context."
    )

    def build(
        self,
        query: str,
        ranked_products: List[RankedProduct],
        preferences: Optional[UserPreferences] = None,
        top_n: int = 3,
    ) -> str:
        context_block = self._build_context(ranked_products[:top_n])
        preference_block = self._build_preferences(preferences)
        return (
            f"{self.system_prompt}\n\n"
            f"Customer request: {query}\n"
            f"{preference_block}"
            f"Candidate products (ranked by relevance):\n{context_block}\n\n"
            "Task: Recommend the best 1-3 products from the candidates above, "
            "with a one-sentence reason for each grounded in the product "
            "description and reviews."
        )

    def _build_preferences(self, preferences: Optional[UserPreferences]) -> str:
        if preferences is None:
            return ""
        parts = []
        if preferences.category:
            parts.append(f"preferred category: {preferences.category}")
        if preferences.max_price is not None:
            parts.append(f"budget: up to ${preferences.max_price:.0f}")
        if preferences.style_notes:
            parts.append(f"style notes: {preferences.style_notes}")
        if not parts:
            return ""
        return "Customer preferences: " + "; ".join(parts) + "\n"

    def _build_context(self, ranked_products: List[RankedProduct]) -> str:
        """Pack the top candidates into the prompt, trimming to a character budget.

        This is the "prompt optimization" step: rather than dumping every
        retrieved review verbatim, each candidate contributes one compact
        summary line, and packing stops once the budget is used up so prompt
        size (and inference cost) stays predictable regardless of catalog size.
        """
        lines = []
        used = 0
        for rp in ranked_products:
            product = rp.product
            snippet = _shorten(product.description, 160)
            review_snippet = _top_review_snippet(product)
            rating = f"{product.average_rating:.1f}" if product.average_rating else "n/a"
            line = (
                f"- {product.name} (${product.price:.0f}, {product.category}, "
                f"avg rating {rating}): {snippet} Review highlight: {review_snippet}"
            )
            if used + len(line) > MAX_CONTEXT_CHARS:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)


def _shorten(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _top_review_snippet(product: Product) -> str:
    if not product.reviews:
        return "no reviews yet"
    best = max(product.reviews, key=lambda r: r.rating)
    return f'"{_shorten(best.text, 120)}" ({best.rating}/5)'
