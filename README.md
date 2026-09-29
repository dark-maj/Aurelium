# Aurelium
An end-to-end AI Research Copilot powered by SPECTER/SciBERT embeddings and RAG. Automates scientific paper ingestion, contextual search, novelty scoring, citation impact prediction, and hypothesis generation.

Machine Learning

---

## What this is

Aurelium takes a pile of research papers and turns them into something you can actually reason over. Everything hangs off one idea: if you embed papers well, a lot of hard questions ("is this new?", "will this get cited?", "what's heating up?") become geometry problems in vector space.

This README covers only the ML side: data, embeddings, the four models, and the RAG layer. Nothing here about servers or UI.

## The pipeline

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/d2e2d82a-4e20-497f-af32-7e08dceca3e2" />



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

## 6. AI Research Copilot (RAG)

The copilot is a retrieval-augmented LLM sitting on top of everything above.

1. Embed the user's question
2. Retrieve relevant papers (and chunks of their abstracts) from the index
3. Attach the ML insights (novelty, predicted impact, topic trend) to each retrieved paper
4. Feed all of it to the LLM as grounded context
5. Generate an answer that cites the papers it used

What it can do:

- **RAG Q&A**: answers grounded in retrieved papers
- **Summarization**: single papers or a whole set
- **Comparison**: side-by-side of methods, results, and assumptions
- **Research gaps**: looks at sparse regions of a topic cluster plus what retrieved papers say is unresolved
- **Hypothesis generation**: proposes ideas by combining findings across neighboring papers, then scores them with the novelty model

Answers should always point back to source papers. If retrieval comes back weak, the copilot should say so instead of making things up.

## Evaluation

| Task | How we check it |
|---|---|
| Retrieval | Recall@k, MRR, nDCG on a held-out set of query/paper pairs |
| Novelty | Correlation with expert-labeled novelty (small hand-labeled set) |
| Citation prediction | Spearman correlation, MAE on log citations, time-based split |
| Trend forecasting | Backtesting: forecast past months, compare to what happened |
| RAG | Faithfulness to sources, answer relevance, manual spot checks |

*Results table coming once the full runs finish.*

## Known limitations

- Citation prediction is inherently noisy. Timing, authors' fame, and plain luck matter a lot and the model can't see them.
- Novelty in embedding space is not the same as scientific novelty. A paper can be far from everything else because it's unusual, or because it's wrong.
- SPECTER only sees title + abstract, so anything buried in the full text is invisible.
- Trend forecasts get shaky on small or brand-new topics.
- The LLM can still hallucinate, even with retrieval. Always check the cited papers.

## Tech used (ML side)

`PyTorch` · `Hugging Face Transformers` · `SPECTER` · `SciBERT` · `FAISS` · `scikit-learn` · `XGBoost / LightGBM` · `UMAP` · `HDBSCAN` · `Prophet` · `sentence-transformers`


