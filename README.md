# Aurelium

An end-to-end AI Research Copilot powered by SPECTER/SciBERT embeddings and RAG. Automates scientific paper ingestion, contextual search, novelty scoring, citation impact prediction, trend forecasting, and hypothesis generation.

Machine Learning · LLM Agents · RAG

---

## What this is

Aurelium takes a pile of research papers and turns them into something you can actually reason over. Enter a topic and get ranked papers, citation-grounded summaries, comparison tables, novelty scores, a trend graph, and suggested research gaps with testable hypotheses.

The project has two halves that meet at a small, well-defined interface:

| Half | Owns | Section |
|---|---|---|
| **ML side** | Data, embeddings, relevance ranking, novelty, citation prediction, trend forecasting | [ML side](#ml-side) |
| **AI side** | Ingestion pipeline, RAG, LLM agents, orchestration, API, LLM evaluation | [AI side](#ai-side) |

The ML side produces numeric signals. The AI side retrieves, reasons, and writes, and reads those signals as grounded context.

## System overview

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/d2e2d82a-4e20-497f-af32-7e08dceca3e2" />

```
topic
  |
  v
query planner --> retriever --> ranker (ML) --> summarizer --> comparator
                                                                   |
                          trend series (ML) <-- novelty scores (ML) |
                                                                   v
                                          gap analyzer --> hypothesis generator <--> critic
                                                                   |
                                                                   v
                                                    ranked papers, summaries, comparison table,
                                                    novelty scores, trend graph, gaps, hypotheses
```

## The ML/AI contract

The two halves talk through four interfaces. Each has a stub implementation returning dummy values, so the AI pipeline runs end to end before the real models are plugged in, and swapping a stub for a real model is a one-line change.

| Interface | Signature | Implemented by |
|---|---|---|
| `RankerProtocol` | `rank(query, papers) -> scores` | Retrieval + cross-encoder rerank |
| `NoveltyProtocol` | `score(paper) -> float` | Novelty detection |
| `ImpactProtocol` | `predict(paper) -> float` | Citation prediction |
| `TrendProtocol` | `forecast(topic) -> series` | Trend forecasting |

If the ML side runs as a separate service, these map to HTTP endpoints described in the OpenAPI spec under `ml_interfaces/`.

---

# ML side

This section covers data, embeddings, the four models, and the ML-side retrieval layer. Nothing here about servers or UI.

## The pipeline

Everything hangs off one idea: if you embed papers well, a lot of hard questions ("is this new?", "will this get cited?", "what's heating up?") become geometry problems in vector space.

## 1. Data collection

Papers come from three places:

- **arXiv** for preprints (CS, stat, physics, etc.)
- **Semantic Scholar** for abstracts, citation counts and the citation graph
- **PubMed** for biomedical work

Each paper ends up as a record with title, abstract, authors, venue, publication date, fields of study, and citation count (where available). We work mostly from title + abstract. Full text is messy, long, and in our experience doesn't add much for the tasks below.

Papers get deduplicated across sources, mainly by DOI, then by fuzzy title match for the ones with no DOI.

## 2. Cleaning & preprocessing

Nothing fancy, but it matters more than you'd expect:

- Strip LaTeX leftovers, HTML tags, weird unicode
- Drop papers with missing or very short abstracts (under ~50 words)
- Normalize whitespace and casing where it's safe to (we keep the original casing for the embedding models, since SciBERT is cased-aware in places)
- Format input as `title [SEP] abstract`, which is what SPECTER expects
- Keep the publication date around. Later steps need it to avoid leaking the future into the past

## 3. Embeddings

This is the core of the project.

- **SPECTER** is our default. It was trained on citation relationships, so papers that cite each other end up close together. That's what we want for search and novelty.
- **SciBERT** is the alternative / baseline, and we use it when we need token-level features or want to fine-tune on a specific domain.

Every paper becomes a 768-dimensional vector. We normalize them so cosine similarity is just a dot product, and store them in a FAISS index for fast nearest-neighbor lookup.

Embeddings are computed once and cached. Re-embedding the whole corpus is slow, so we only run new papers through the model.

## 4. The four tasks

All four share the same embeddings. That's deliberate: one representation, several heads.

### Information retrieval
Given a query (a question, a paper title, or a chunk of text), embed it and pull the top-k nearest papers from FAISS. We then rerank the candidates with a cross-encoder for better precision at the top of the list. This is also the retrieval step that feeds the RAG layer.

### Novelty detection
How different is a paper from what already exists? We look at its neighborhood in embedding space: a paper whose nearest neighbors are all far away is more novel than one sitting in a dense cluster.

Roughly: `novelty = 1 - mean cosine similarity to its k nearest neighbors`, computed against papers published *before* it. That last part is important. Otherwise you punish early papers for being followed by lots of similar ones.

We also tried Local Outlier Factor as a comparison.

### Citation prediction
Predict how many citations a paper will get, using its embedding plus a few metadata features (number of authors, venue, field, novelty score). Targets are log-transformed since citation counts are heavily skewed.

We use gradient boosting as the main model, with a small MLP as a comparison. Train/test splits are **by time**, never random. A random split lets the model peek at the future and the numbers look great for the wrong reasons.

### Research trend forecasting
1. Cluster the embeddings (UMAP for dimensionality reduction, then HDBSCAN) to find topics.
2. Count papers per topic per month.
3. Forecast each topic's volume with a time-series model (Prophet / ARIMA).

Topics whose forecasted growth stands out get flagged as emerging.

## 5. ML insights layer

The four models each produce a signal. This layer combines them into one summary per paper or topic:

| Signal | Comes from |
|---|---|
| Relevance | Retrieval |
| Novelty | Novelty detection |
| Citation impact | Citation prediction |
| Emerging trends | Trend forecasting |

These are what the copilot reads when it answers something like "what's worth reading in this area?"

## ML evaluation

| Task | How we check it |
|---|---|
| Retrieval | Recall@k, MRR, nDCG on a held-out set of query/paper pairs |
| Novelty | Correlation with expert-labeled novelty (small hand-labeled set) |
| Citation prediction | Spearman correlation, MAE on log citations, time-based split |
| Trend forecasting | Backtesting: forecast past months, compare to what happened |

## ML known limitations

- Citation prediction is inherently noisy. Timing, authors' fame, and plain luck matter a lot and the model can't see them.
- Novelty in embedding space is not the same as scientific novelty. A paper can be far from everything else because it's unusual, or because it's wrong.
- SPECTER only sees title + abstract, so anything buried in the full text is invisible.
- Trend forecasts get shaky on small or brand-new topics.

## Tech used (ML side)

`PyTorch` · `Hugging Face Transformers` · `SPECTER` · `SciBERT` · `FAISS` · `scikit-learn` · `XGBoost / LightGBM` · `UMAP` · `HDBSCAN` · `Prophet` · `sentence-transformers`

---

# AI side

This section covers ingestion, the RAG layer, the LLM agents, orchestration, the API, and LLM evaluation.

## Architecture

| Layer | Responsibility |
|---|---|
| Ingestion | Fetch, deduplicate, and store papers and their metadata |
| Retrieval | Hybrid search over chunks with citation-verifiable offsets |
| Agents | Summarize, compare, find gaps, generate and critique hypotheses |
| Orchestration | LangGraph workflow with per-run state and streamed progress |
| API | FastAPI service consumed by the frontend |
| Evaluation | Retrieval, faithfulness, and hypothesis-quality checks |

Default stack: Python 3.11, FastAPI, LangGraph, Postgres with pgvector, Pydantic v2.

### Repository layout

```
ingestion/
retrieval/
agents/
api/
ml_interfaces/
eval/
```

## 1. Ingestion

Fetches papers by topic from arXiv and Semantic Scholar (title, abstract, authors, year, venue, citation count, references, external IDs) and writes them to Postgres.

- Async `httpx` clients with rate limiting and exponential backoff
- Deduplication across sources by DOI, then arXiv ID, then normalized title
- Chunking is abstract-level by default, matching the ML side's title + abstract focus; section-aware chunking of full text is an optional mode when a PDF is available
- Embedding model is configurable and should match the ML side's SPECTER default so both halves agree on the representation
- Chunk embeddings live in pgvector for RAG; the paper-level FAISS index stays owned by the ML side

```
python -m ingestion.run --topic "your topic" --limit 200
```

Tests use mocked API responses.

## 2. Retrieval (RAG)

- Hybrid search: dense vectors plus BM25 via Postgres full-text
- Reciprocal rank fusion to merge the two result lists
- Optional cross-encoder rerank
- A hook where `RankerProtocol` re-ranks the final candidates
- Every returned chunk carries `paper_id`, section, and character offsets so citations can be verified
- Retrieval parameters are configurable and per-stage latency is logged
- If retrieval quality is weak, the pipeline says so instead of proceeding to generation

## 3. Agents

Every agent uses only retrieved context, returns JSON validated by Pydantic, and uses schema-constrained output rather than free-text parsing.

| Agent | What it does |
|---|---|
| Query planner | Expands a topic into 4-6 diverse sub-queries (core methods, benchmarks, surveys, recent work, adjacent fields) |
| Summarizer | Structured summary (problem, method, datasets, results, limitations) with a chunk citation on every claim |
| Verifier | Second pass that checks each claim against its cited chunk and removes or flags unsupported ones |
| Comparator | Picks comparison dimensions dynamically and fills a table; each cell has a citation and confidence, and "not reported" is valid |
| Gap analyzer | Finds uncovered method/dataset combinations, recurring limitations, and contradictory findings, using novelty scores and the comparison table |
| Hypothesis generator | Up to 5 hypotheses, each with motivation, concrete experiment (dataset, baseline, metric), expected outcome, and risks |
| Critic | Scores groundedness, testability, and novelty 1-5, checks against retrieved papers, and sends weak hypotheses back for revision (max 2 rounds) |

## 4. Orchestration

A LangGraph workflow wires the agents together:

`query_planner -> retriever -> ranker -> summarizer -> comparator -> gap_analyzer -> hypothesis_generator <-> critic`

State is persisted per run, and progress events are streamed to the client over SSE.

## 5. API

| Endpoint | Purpose |
|---|---|
| `POST /research` | Submit a topic, get a job ID |
| `GET /research/{id}` | Status and results |
| `GET /research/{id}/stream` | SSE progress stream |

The result payload contains ranked papers, summaries, the comparison table, novelty scores, trend series, suggested gaps, and hypotheses. The service includes CORS, request validation, and structured logging.

## AI evaluation

| Task | How we check it |
|---|---|
| Retrieval | nDCG@10 and Recall@100 on SciFact/BEIR for dense, BM25, hybrid, and reranked configurations |
| Summary faithfulness | LLM judge on 50 sampled summaries, plus manual spot check |
| Comparison accuracy | Manual check of cell values and citations on a small sample |
| Hypothesis quality | LLM-judge rubric for groundedness, testability, and novelty |

*Results table coming once the full runs finish.*

## AI known limitations

- The LLM can still hallucinate, even with retrieval. Always check the cited papers.
- Citation verification catches unsupported claims but not claims that are supported by a wrong or low-quality source.
- Hypotheses are starting points for a researcher, not validated findings. Novelty is judged against retrieved papers only, so something may already exist outside the corpus.
- Abstract-level chunks limit how much detail summaries and comparisons can contain.

## Tech used (AI side)

`Python` · `FastAPI` · `LangGraph` · `Pydantic` · `Postgres` · `pgvector` · `httpx` · `sentence-transformers`

---

## Overall status

Results for both halves are pending the full evaluation runs.
