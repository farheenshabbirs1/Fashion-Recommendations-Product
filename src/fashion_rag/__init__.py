"""fashion_rag: an LLM-powered fashion product recommendation engine.

Pipeline: embedding-based semantic retrieval -> contextual re-ranking ->
prompt construction -> LLM generation. See the package README for the
full architecture and usage.
"""

__version__ = "0.1.0"

from .models import Product, Review, UserPreferences  # noqa: F401
from .pipeline import RecommendationPipeline, RecommendationResult  # noqa: F401
