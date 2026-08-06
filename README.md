# AI Product Manager

An evidence-based product-feedback pipeline for Google Play applications. It
uses deterministic collection and scoring, TF-IDF + XGBoost classification,
local retrieval for market evidence, and an optional Gemini call for report
writing.

## Current pipeline

```text
Exact app identification
  → review collection
  → batched category classification + confidence
  → sentiment + near-duplicate grouping + priority scoring
  → feature opportunity ranking
  → app metadata + competitor evidence retrieval
  → Markdown report (Gemini or offline fallback)
```

The LLM writes and synthesizes; it does not classify reviews or calculate
priority.

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Export `GOOGLE_API_KEY` to enable Gemini report synthesis. Without a key, the
pipeline produces a deterministic offline Markdown report.

## Run

CLI:

```bash
python3 main.py "Instagram" --count 200
```

For an app not present in the verified alias map, pass the package explicitly:

```bash
python3 main.py "My App" --app-id com.example.myapp --count 200
```

Streamlit:

```bash
streamlit run streamlit_app.py
```

FastAPI:

```bash
uvicorn AI_Product_Manager.api.app:app --reload
```

Then send:

```json
POST /analyze
{"app_name": "Instagram", "review_count": 200}
```

## Outputs

- Raw reviews: `AI_Product_Manager/data/raw_reviews/`
- Analyzed runs: `AI_Product_Manager/data/runs/`
- Reports: `reports/`

## Human review

- Non-exact app matches return candidate applications instead of silently
  selecting one.
- Reviews below `APM_CLASSIFICATION_THRESHOLD` are marked
  `needs_human_review`.
- Competitor seed data is explicitly treated as unverified unless supported by
  a document in `AI_Product_Manager/rag/knowledge/`.

## Local RAG knowledge

Add verified competitor descriptions, release notes, and changelogs as `.md` or
`.txt` documents under `AI_Product_Manager/rag/knowledge/`. Include source URLs
and retrieval dates. Documents are persisted in ChromaDB using a local,
no-download hashing embedding.

Collect competitor Play Store descriptions and recent changes, then ingest:

```bash
python3 -c "from AI_Product_Manager.tools.competitor_research import collect_competitor_knowledge; print(collect_competitor_knowledge('Instagram'))"
python3 -m AI_Product_Manager.rag.ingest
```

## Workflow state and human review

LangGraph executes classification, sentiment, deduplication, confidence gating,
and priority scoring. SQLite stores run state in
`AI_Product_Manager/data/pipeline_state.sqlite3`. Low-confidence rows are saved
under `AI_Product_Manager/data/human_review/`.

Background API workflow:

```text
POST /jobs
GET  /jobs/{run_id}
POST /jobs/{run_id}/review
```

The review endpoint records approval/rejection and PM notes. Both Markdown and
PDF reports are produced.

## Tests

```bash
pytest -q
```

## Reproducible model training

The included model pickle is retained for compatibility. Retrain it in the
current environment to remove cross-version XGBoost serialization warnings:

```bash
python3 -m AI_Product_Manager.ml.train_classifier \
  --dataset AI_Product_Manager/data/final_labeled_feedback.csv
```

This saves the model, vectorizer, label encoder, and evaluation metrics under
`AI_Product_Manager/models/`.

## Known limitations

- Google Play scraping depends on an unofficial external library and network
  availability.
- Lexicon sentiment is directional, not a nuanced language model.
- TF-IDF duplicate grouping is a local fallback; semantic embeddings improve
  paraphrase matching.
- Root causes cannot be proven from reviews alone and require telemetry or
  engineering investigation.
# multi-agent-product-management-system
