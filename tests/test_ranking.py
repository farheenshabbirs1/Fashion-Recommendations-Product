from fashion_rag.models import Product, Review, UserPreferences
from fashion_rag.ranking import ContextualRanker
from fashion_rag.retrieval import RetrievedProduct


def _retrieved(id_: str, category: str, price: float, similarity: float, rating: int) -> RetrievedProduct:
    product = Product(id=id_, name=id_, category=category, price=price, tags=[], description="")
    product.reviews = [Review(product_id=id_, rating=rating, text="great")]
    return RetrievedProduct(product=product, similarity=similarity)


def test_ranker_prefers_matching_category_and_budget_over_raw_similarity():
    ranker = ContextualRanker()
    candidates = [
        _retrieved("expensive_wrong_category", "shoes", 500, similarity=0.75, rating=5),
        _retrieved("on_budget_right_category", "dresses", 60, similarity=0.7, rating=4),
    ]
    preferences = UserPreferences(category="dresses", max_price=100)

    ranked = ranker.rank(candidates, preferences)

    assert ranked[0].product.id == "on_budget_right_category"


def test_ranker_falls_back_to_rating_without_preferences():
    ranker = ContextualRanker()
    candidates = [
        _retrieved("low_rated", "tops", 40, similarity=0.5, rating=2),
        _retrieved("high_rated", "tops", 40, similarity=0.5, rating=5),
    ]

    ranked = ranker.rank(candidates, preferences=None)

    assert ranked[0].product.id == "high_rated"
