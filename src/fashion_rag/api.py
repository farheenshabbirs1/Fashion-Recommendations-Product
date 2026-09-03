"""FastAPI wrapper around ``RecommendationPipeline``.

The rest of this package is a library -- ``scripts/run_demo.py`` calls it directly from a CLI
process that exits when it's done. This module turns the same pipeline into a small, long-running
HTTP service, which is what a container, a Kubernetes Deployment/Service, or a blue-green
rollout actually needs something to target. Nothing about the recommendation logic changes here;
this is only a thin request/response boundary around ``RecommendationPipeline.recommend``.

Run directly:
    uvicorn fashion_rag.api:app --reload

Or via the Docker image (see ../../Dockerfile and ../../infra/docker-compose.yml).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data_loader import load_products
from .models import UserPreferences
from .pipeline import RecommendationPipeline

logger = logging.getLogger("fashion_rag.api")

_pipeline: Optional[RecommendationPipeline] = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    # Build the pipeline (and its embedding index) once at process startup rather than on the
    # first request, so a Kubernetes readiness probe hitting /ready right after boot reflects
    # real state instead of racing the first real request.
    get_pipeline()
    yield


app = FastAPI(
    title="Fashion Recommendations API",
    description=(
        "RAG-based fashion product recommendations: embedding-based retrieval, "
        "contextual ranking, prompt-optimized generation."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


def get_pipeline() -> RecommendationPipeline:
    """Builds the pipeline once per process and reuses it across requests.

    Building indexes the whole bundled catalog (embeds every product), so this should happen
    once at startup rather than per-request -- the ``startup`` hook below does exactly that so a
    Kubernetes readiness probe hitting ``/ready`` right after boot reflects real state instead of
    racing the first real request.
    """
    global _pipeline
    if _pipeline is None:
        embedder_name = os.environ.get("EMBEDDER")
        llm_provider = os.environ.get("LLM_PROVIDER")
        logger.info(
            "Building recommendation pipeline (embedder=%s, llm=%s)",
            embedder_name or "tfidf (default)",
            llm_provider or "mock (default)",
        )
        products = load_products()
        _pipeline = RecommendationPipeline.build(
            products, embedder_name=embedder_name, llm_provider=llm_provider
        )
    return _pipeline


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text shopper request.")
    category: Optional[str] = Field(None, description="Preferred product category.")
    max_price: Optional[float] = Field(None, ge=0, description="Budget ceiling in USD.")
    top_k: int = Field(8, ge=1, le=50, description="Candidates to retrieve before ranking.")
    top_n_in_prompt: int = Field(
        3, ge=1, le=10, description="Ranked candidates to pass into the generation prompt."
    )


class RankedProductOut(BaseModel):
    id: str
    name: str
    category: str
    price: float
    similarity: float
    context_score: float
    final_score: float


class RecommendResponse(BaseModel):
    query: str
    ranked_products: List[RankedProductOut]
    generated_text: str


@app.get("/health")
def health() -> dict:
    """Liveness probe target: the process is up and serving, independent of pipeline state."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness probe target: the pipeline (and its embedding index) is built and queryable."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not built yet")
    return {"status": "ready"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    pipeline = get_pipeline()
    preferences = UserPreferences(category=request.category, max_price=request.max_price)
    result = pipeline.recommend(
        request.query,
        preferences=preferences,
        top_k=request.top_k,
        top_n_in_prompt=request.top_n_in_prompt,
    )
    return RecommendResponse(
        query=result.query,
        ranked_products=[
            RankedProductOut(
                id=rp.product.id,
                name=rp.product.name,
                category=rp.product.category,
                price=rp.product.price,
                similarity=rp.similarity,
                context_score=rp.context_score,
                final_score=rp.final_score,
            )
            for rp in result.ranked_products
        ],
        generated_text=result.generated_text,
    )
