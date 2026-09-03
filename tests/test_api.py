"""Tests for the FastAPI wrapper (``fashion_rag.api``)."""
from fastapi.testclient import TestClient

from fashion_rag.api import app


def test_health_does_not_require_pipeline():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_after_startup():
    # TestClient runs FastAPI startup/shutdown hooks when used as a context manager, so the
    # pipeline (and its embedding index) is built before this request goes out.
    with TestClient(app) as client:
        resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_recommend_returns_ranked_products_and_generated_text():
    with TestClient(app) as client:
        resp = client.post(
            "/recommend",
            json={"query": "something breathable and casual for a beach vacation"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "something breathable and casual for a beach vacation"
    assert len(body["ranked_products"]) > 0
    top = body["ranked_products"][0]
    assert {"id", "name", "category", "price", "similarity", "context_score", "final_score"} <= top.keys()
    assert isinstance(body["generated_text"], str) and body["generated_text"]


def test_recommend_applies_preferences():
    with TestClient(app) as client:
        resp = client.post(
            "/recommend",
            json={"query": "office attire", "category": "bottoms", "max_price": 100},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["ranked_products"]) > 0


def test_recommend_rejects_empty_query():
    with TestClient(app) as client:
        resp = client.post("/recommend", json={"query": ""})
    assert resp.status_code == 422
