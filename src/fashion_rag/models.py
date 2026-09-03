"""Core data models for the fashion recommendation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Review:
    product_id: str
    rating: int
    text: str


@dataclass
class Product:
    id: str
    name: str
    category: str
    price: float
    tags: List[str]
    description: str
    reviews: List[Review] = field(default_factory=list)

    @property
    def average_rating(self) -> Optional[float]:
        if not self.reviews:
            return None
        return sum(r.rating for r in self.reviews) / len(self.reviews)

    def to_document(self) -> str:
        """Flatten the product's fields and reviews into one document for embedding."""
        review_text = " ".join(r.text for r in self.reviews)
        tag_text = ", ".join(self.tags)
        return (
            f"{self.name}. Category: {self.category}. Tags: {tag_text}. "
            f"{self.description} Customer reviews: {review_text}"
        ).strip()


@dataclass
class UserPreferences:
    """Optional shopper context used to bias ranking, independent of the query text."""

    category: Optional[str] = None
    max_price: Optional[float] = None
    style_notes: Optional[str] = None
