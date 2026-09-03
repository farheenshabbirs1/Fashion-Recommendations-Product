from fashion_rag.embeddings import TfidfEmbedder
from fashion_rag.llm_client import MockLLMClient
from fashion_rag.models import Product
from fashion_rag.pipeline import RecommendationPipeline
from fashion_rag.retrieval import SemanticRetriever


def test_pipeline_end_to_end_with_mock_llm():
    products = [
        Product(
            id="a",
            name="Linen Beach Dress",
            category="dresses",
            price=70,
            tags=["linen", "beach"],
            description="A breathable linen dress for beach days.",
        ),
        Product(
            id="b",
            name="Wool Winter Coat",
            category="outerwear",
            price=150,
            tags=["wool", "winter"],
            description="A heavy wool coat for cold winter weather.",
        ),
    ]
    retriever = SemanticRetriever(embedder=TfidfEmbedder())
    retriever.index(products)
    pipeline = RecommendationPipeline(retriever=retriever, llm_client=MockLLMClient())

    result = pipeline.recommend("something for a beach vacation")

    assert result.ranked_products[0].product.id == "a"
    assert "Linen Beach Dress" in result.generated_text


def test_pipeline_end_to_end_on_full_bundled_catalog():
    from fashion_rag.data_loader import load_products

    products = load_products()
    pipeline = RecommendationPipeline.build(products)

    result = pipeline.recommend("warm cozy outfit for cold winter weather")

    top_ids = {rp.product.id for rp in result.ranked_products[:5]}
    assert top_ids & {"p005", "p017", "p019", "p025"}
    assert result.generated_text
