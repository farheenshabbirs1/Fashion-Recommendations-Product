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
│   └── evaluate.py         # precision/recall/F1 vs. a keyword baseline
├── scripts/
│   ├── run_demo.py         # CLI: run one query end to end
│   └── run_evaluation.py   # CLI: report retrieval quality
├── tests/                  # pytest suite (retrieval, ranking, pipeline)
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

  system     precision=0.600  recall=0.975  f1=0.717
  baseline   precision=0.500  recall=0.797  f1=0.593

Embedding-based retrieval improves F1 by 21.0% over the keyword baseline.
```

Numbers will shift slightly if you edit the bundled catalog or eval set, since
they're computed live rather than hardcoded -- this run's ~21% F1 lift is in
the same range as the recommendation-quality improvement cited on the resume
bullet this project backs.

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
behavior with and without preferences (`test_ranking.py`), and the full
pipeline end to end against both a tiny synthetic catalog and the bundled
dataset (`test_pipeline.py`).
