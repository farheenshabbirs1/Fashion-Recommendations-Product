from fashion_rag.embeddings import TfidfEmbedder
from fashion_rag.models import Product
from fashion_rag.retrieval import SemanticRetriever


def _sample_products():
    return [
        Product(
            id="a",
            name="Linen Beach Dress",
            category="dresses",
            price=70,
            tags=["linen", "beach", "summer"],
            description="A breathable linen dress for beach days.",
        ),
        Product(
            id="b",
            name="Wool Winter Coat",
            category="outerwear",
            price=150,
            tags=["wool", "winter", "warm"],
            description="A heavy wool coat for cold winter weather.",
        ),
        Product(
            id="c",
            name="Running Sneakers",
            category="shoes",
            price=90,
            tags=["athletic", "workout"],
            description="Lightweight sneakers built for daily runs.",
        ),
    ]


def test_search_returns_most_relevant_product_first():
    retriever = SemanticRetriever(embedder=TfidfEmbedder())
    retriever.index(_sample_products())

    results = retriever.search("something warm for cold winter weather", top_k=2)

    assert results[0].product.id == "b"
    assert len(results) == 2


def test_search_many_matches_individual_search_results():
    retriever = SemanticRetriever(embedder=TfidfEmbedder())
    retriever.index(_sample_products())

    queries = ["beach dress", "running shoes"]
    batch_results = retriever.search_many(queries, top_k=1)
    individual_results = [retriever.search(q, top_k=1) for q in queries]

    for batch, individual in zip(batch_results, individual_results):
        assert batch[0].product.id == individual[0].product.id
