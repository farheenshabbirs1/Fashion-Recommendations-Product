# Fashion Product Recommendations

An LLM-powered product recommendation engine that combines **embedding-based
semantic retrieval**, **contextual ranking**, and **prompt-optimized
generation** to turn a free-text shopping request into grounded, explainable
fashion recommendations.

Given a request like *"something breathable and casual for a beach
vacation"*, the pipeline retrieves the most semantically relevant products
and reviews, re-ranks them using the shopper's stated preferences (category,
budget), builds a compact prompt from the top candidates, and asks an LLM to
generate a short, grounded recommendation.

```
query ─▶ SemanticRetriever ─▶ ContextualRanker ─▶ PromptBuilder ─▶ LLMClient ─▶ recommendation
         (embeddings, top-k)   (preferences,        (budget-aware    (mock /
                                 rating, price)       context pack)    OpenAI / Anthropic)
```

## Why it's built this way

- **Embedding-based retrieval** (`src/fashion_rag/retrieval.py`) indexes every
  product's description, tags, and reviews as one document, embeds the
  catalog **in parallel** (`ThreadPoolExecutor`), and ranks candidates by
  cosine similarity -- this is what lets a query like *"cozy for cold
  weather"* match a "wool sweater dress" that never uses those exact words.
- **Contextual ranking** (`ranking.py`) blends that similarity score with the
  shopper's category preference, budget fit, and average review rating, so
  the top semantic match isn't always the top recommendation.
- **Prompt optimization** (`prompting.py`) packs only the top-N ranked
  candidates into the prompt, trims each to a compact summary line, and stops
  once a character budget is hit -- keeping prompt size, and inference cost,
  predictable regardless of catalog size.
- **Provider-agnostic generation** (`llm_client.py`) defaults to a
  deterministic, template-based `MockLLMClient` so the whole pipeline runs
  with **zero API keys and zero network calls**. Set `LLM_PROVIDER` to switch
  to OpenAI or Anthropic without touching any other code.
- **Retrieval evaluation** (`evaluate.py`) computes precision/recall/F1 for
  the semantic retriever against a naive keyword-overlap baseline on a set of
  hand-labeled queries, to quantify the lift from embedding-based retrieval.

## Project layout

```
fashion-rag-recommender/
├── data/
│   ├── products.json       # 26 synthetic fashion products
│   ├── reviews.json        # customer reviews per product
│   └── eval_queries.json   # labeled queries for retrieval evaluation
├── src/fashion_rag/
│   ├── models.py           # Product, Review, UserPreferences
│   ├── data_loader.py      # loads data/*.json into models
│   ├── embeddings.py       # Embedder: TfidfEmbedder (default), SBERT, OpenAI
│   ├── retrieval.py        # SemanticRetriever: parallel indexing + top-k search
│   ├── ranking.py          # ContextualRanker: similarity + preferences + rating
│   ├── prompting.py        # PromptBuilder: budget-aware context packing
│   ├── llm_client.py       # LLMClient: Mock (default), OpenAI, Anthropic
│   ├── pipeline.py         # RecommendationPipeline: wires it all together
│   ├── api.py              # FastAPI wrapper around the pipeline (see "Running the API")
│   └── evaluate.py         # precision/recall/F1 vs. a keyword baseline
├── scripts/
│   ├── run_demo.py         # CLI: run one query end to end
│   └── run_evaluation.py   # CLI: report retrieval quality
├── tests/                  # pytest suite (retrieval, ranking, pipeline, API)
├── Dockerfile               # builds the API service image
├── infra/
│   ├── docker-compose.yml   # runs the service locally, one command
│   ├── terraform/           # reference Azure infra (AKS + ACR) -- not applied, see "Known limitations"
│   └── k8s/                 # plain Kubernetes manifests + a blue-green rollout example
├── .github/workflows/ci.yml # tests + Docker build on push
├── pyproject.toml
└── requirements.txt
```

## Quickstart

```bash
git clone <your-fork-url>
cd fashion-rag-recommender
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run a query through the full pipeline (no API key needed -- uses the mock LLM)
python scripts/run_demo.py "something breathable and casual for a beach vacation"

# Try preferences
python scripts/run_demo.py "office attire" --category bottoms --max-price 100

# Evaluate retrieval quality vs. a keyword baseline
python scripts/run_evaluation.py

# Run the tests
pytest
```

Sample `run_demo.py` output:

```
Query: something breathable and casual for a beach vacation

Top ranked candidates:
  High-Waisted Linen Shorts       sim=0.265  ctx=0.950  final=0.539
  Espadrille Sandals              sim=0.250  ctx=0.950  final=0.530
  Straw Tote Bag                  sim=0.237  ctx=0.950  final=0.522
  Linen Wrap Midi Dress           sim=0.220  ctx=0.950  final=0.512
  Boho Maxi Dress                 sim=0.146  ctx=0.950  final=0.468

Generated recommendation:

Based on your request, here are my top picks:
- High-Waisted Linen Shorts: Breathable high-waisted linen shorts with a
  relaxed fit, ideal for beach days and warm-weather travel.
- Espadrille Sandals: Comfortable woven espadrille sandals with a low wedge,
  ideal for beach vacations and casual summer outings.
- Straw Tote Bag: A spacious woven straw tote bag, perfect for carrying beach
  essentials on a summer vacation.
```

## Running the API

`scripts/run_demo.py` is a CLI process that exits after one query. `src/fashion_rag/api.py` wraps
the same `RecommendationPipeline` in a small FastAPI service instead, so it can run as a
long-lived, containerized, load-balanced process -- this is what `infra/` (below) actually
deploys.

```bash
pip install -e ".[api]"
uvicorn fashion_rag.api:app --reload

# in another terminal
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "something breathable and casual for a beach vacation"}'
```

`GET /health` is a liveness check (process is up), `GET /ready` is a readiness check (the
embedding index has finished building), and `POST /recommend` takes the same `query` /
`category` / `max_price` / `top_k` / `top_n_in_prompt` parameters as `run_demo.py`'s CLI flags.
Interactive API docs are served at `/docs`.

## Infrastructure

The project ships with the same Docker + Kubernetes + Terraform depth as this portfolio's JobFit
Checker project, adapted to a single stateless service instead of three microservices:

```bash
cd infra
docker compose up --build
```

This builds the image from the root `Dockerfile` and starts the API on http://localhost:8000
with the mock LLM and TF-IDF embeddings -- no API key required. `infra/k8s/` has plain
Kubernetes manifests (namespace, config, a `Deployment`/`Service` pair) plus
`blue-green-example.yaml`, an illustrative two-Deployment blue-green cutover matching the pattern
in JobFit Checker's `infra/k8s/blue-green-example.yaml`. `infra/terraform/` is reference-only
Azure infrastructure (AKS + ACR) describing the target deployment shape -- **not applied against
a real subscription from here** (see "Known limitations" below). `.github/workflows/ci.yml`
runs the test suite and builds the Docker image on every push, no cloud credentials required.

## Switching providers

Nothing else in the code changes -- swap the backend via environment
variables (or the `--embedder` / `--llm` flags on `run_demo.py`):

```bash
# Use OpenAI for both embeddings and generation
export EMBEDDER=openai
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python scripts/run_demo.py "elegant outfit for a date night"

# Use Anthropic for generation, local sentence-transformers for retrieval
pip install -e ".[anthropic,sbert]"
export EMBEDDER=sbert
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_demo.py "elegant outfit for a date night"
```

## Evaluation

`scripts/run_evaluation.py` measures retrieval quality on `data/eval_queries.json`
(10 hand-labeled queries) and compares the semantic retriever against a naive
keyword-overlap baseline:

```
Retrieval quality (averaged over labeled eval queries):

  system     precision=0.700  recall=0.922  f1=0.766
  baseline   precision=0.600  recall=0.777  f1=0.652

Embedding-based retrieval improves F1 by 17.5% over the keyword baseline.
```

Numbers will shift slightly if you edit the bundled catalog, eval set, or the
`top_k` the evaluation is run at (currently top-4, set in
`scripts/run_evaluation.py`), since they're computed live rather than
hardcoded -- this run's ~17.5% F1 lift matches the recommendation-quality
improvement cited on the resume bullet this project backs.

## Known limitations

- **Terraform is reference-only.** `infra/terraform/` (`main.tf`, `variables.tf`, `outputs.tf`)
  describes an AKS + ACR deployment shape but has not been applied against a real Azure
  subscription from here -- run `terraform plan` yourself against your own subscription before
  ever running `apply`. It doesn't cover the blue-green cutover mechanics (that's a
  Kubernetes-level concern -- see `infra/k8s/blue-green-example.yaml`), secrets management (the
  default config needs no API keys at all; a real provider's key would be a Kubernetes Secret,
  not Terraform/Key-Vault-backed here), or autoscaling (`node_count` and `replicas` are both
  fixed).
- **Blue-green is illustrated, not wired into CI.** `infra/k8s/blue-green-example.yaml` shows the
  two-Deployment-plus-selector-flip pattern; the CI workflow builds and tests the image but
  doesn't deploy it, since there's no cluster or registry connected to this project to deploy to.
- **This sandbox could not verify the Docker build against a running daemon.** The `Dockerfile`
  and `docker-compose.yml` were written and the underlying `pip install -e ".[api]"` path was
  verified in a plain virtualenv (module imports cleanly, all FastAPI routes register), but no
  Docker daemon was reachable from the environment this was built in to run `docker build`
  end to end. Run it yourself (`cd infra && docker compose up --build`) before relying on the
  image, or let the CI workflow's `docker` job do it on push.
- **One stateless service, not three.** Unlike JobFit Checker's multi-service, event-driven
  design, this project is a single Python package with no persistence layer, so its infra mirrors
  JobFit Checker's *structure* (Docker, Kubernetes, reference Terraform, blue-green example) at a
  scale that actually fits a synchronous retrieve-rank-prompt-generate pipeline -- there's no
  outbox, message queue, or database here to add infra around.

## Extending beyond the bundled sample data

The bundled catalog in `data/` is a small synthetic dataset so the project
clones and runs immediately with no downloads or API keys. To point it at a
real catalog, replace `data_loader.load_products()` with a loader for your
own source (a database, a CSV export, a real reviews dataset) that returns
`Product` objects -- everything downstream (retrieval, ranking, prompting,
generation, evaluation) is unchanged.

## Tests

```bash
pytest -v
```

Covers: retrieval correctness (`test_retrieval.py`), contextual re-ranking
behavior with and without preferences (`test_ranking.py`), the full
pipeline end to end against both a tiny synthetic catalog and the bundled
dataset (`test_pipeline.py`), and the FastAPI wrapper's health/readiness
checks and `/recommend` endpoint (`test_api.py`, needs `pip install -e ".[api,dev]"`).
