# Runs the FastAPI wrapper (src/fashion_rag/api.py) around RecommendationPipeline.
# Defaults to the mock LLM and TF-IDF embedder, so the image serves real recommendations with
# zero API keys and zero network calls -- override EMBEDDER / LLM_PROVIDER (+ the matching
# *_API_KEY) at runtime to switch backends, see infra/docker-compose.yml and README.md.

FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies first so the layer is cached across code-only changes.
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e ".[api]"

COPY src/ ./src/
COPY data/ ./data/

RUN useradd --create-home --uid 1000 fashionrag
USER fashionrag

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "fashion_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
