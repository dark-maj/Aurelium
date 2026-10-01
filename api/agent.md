# AGENTS.md

Instructions for AI coding agents working in the Aurelium monorepo. Read this file fully before making any change. If a human instruction conflicts with this file, follow the human and mention the conflict.

## Project summary

Aurelium is an AI Research Copilot. A user enters a topic and gets ranked papers, citation-grounded summaries, comparison tables, novelty scores, a trend graph, and suggested research gaps with hypotheses.

The repo has two halves with separate owners:

- `ml/` holds data processing, embeddings, relevance ranking, novelty detection, citation prediction, and trend forecasting.
- `ai/` holds ingestion, RAG retrieval, LLM agents, LangGraph orchestration, the API, and LLM evaluation.

The two halves communicate only through the interfaces in `contracts/`.

## Repository structure

```
aurelium/
  AGENTS.md
  README.md
  contracts/
    schemas/
    interfaces/
    openapi/
  ai/
    ingestion/
    retrieval/
    agents/
    orchestration/
    api/
    eval/
    tests/
    pyproject.toml
  ml/
    data/
    embeddings/
    models/
    insights/
    eval/
    tests/
    pyproject.toml
  web/
  scripts/
  .github/
```

Rules for the structure:

- Do not create new top-level directories without being asked.
- Each of `ai/` and `ml/` is an independent Python package with its own `pyproject.toml`, dependencies, tests, and config.
- `web/` is reserved for the frontend. Leave it alone unless the task is explicitly about the frontend.
- Shared Pydantic schemas and protocol definitions live only in `contracts/`. Never duplicate them inside `ai/` or `ml/`.

## Ownership and boundaries

- Work only inside the directory your task belongs to. An agent working on `ai/` must not edit files in `ml/`, and the reverse.
- `ai/` must never import from `ml/`, and `ml/` must never import from `ai/`. Both may import from `contracts/`.
- Changes to anything in `contracts/` affect both owners. Keep them backward compatible where possible, update both sides' stubs and tests, and call out the change clearly in the PR description so both owners review it.
- If a task needs a change on the other side, stop and write down what is needed instead of making it.

## The ML/AI contract

The four interfaces below are the only supported way for the AI side to call ML functionality:

| Interface | Signature | Provided by |
|---|---|---|
| `RankerProtocol` | `rank(query, papers) -> scores` | `ml/` |
| `NoveltyProtocol` | `score(paper) -> float` | `ml/` |
| `ImpactProtocol` | `predict(paper) -> float` | `ml/` |
| `TrendProtocol` | `forecast(topic) -> series` | `ml/` |

- Stub implementations returning dummy values live in `contracts/interfaces/` so each side can run and test without the other.
- Real implementations are selected by configuration, not by editing call sites.
- If ML runs as a separate service, the HTTP shape is defined in `contracts/openapi/` and must stay in sync with the protocols.

## Rules for ML work (`ml/`)

- Never leak the future. Splits for citation prediction and trend forecasting are by time, never random. Novelty is computed only against papers published before the paper being scored.
- Keep the publication date on every paper record through the whole pipeline.
- Use title + abstract in the `title [SEP] abstract` format for SPECTER unless told otherwise. Embeddings are normalized so cosine similarity is a dot product.
- Cache embeddings and only embed new papers.
- Seed every source of randomness and record the seed, model version, and data snapshot with each reported result.
- Never report a metric that was not actually computed. If a result is missing, write "pending".
- Do not commit datasets, embedding caches, FAISS indexes, or model weights. Document how to regenerate them.

## Rules for AI work (`ai/`)

- Agents use only retrieved context. Every factual claim in a generated summary, comparison, gap, or hypothesis must carry a citation to a specific chunk or paper ID.
- Never invent papers, numbers, datasets, or results. "Not reported" is always a valid output.
- LLM outputs that feed other code must use schema-constrained output validated by Pydantic. Do not parse free text with regex.
- Retrieval returns chunks with `paper_id`, section, and character offsets so citations can be verified.
- If retrieval quality is weak, the system must say so instead of generating a confident answer.
- Keep prompts in versioned files, not scattered inline strings, and keep model names and parameters in configuration.
- All external API calls (arXiv, Semantic Scholar, PubMed, LLM providers) use async clients with timeouts, retries, and backoff.

## Coding standards

- Python 3.11, type hints on all public functions, Pydantic v2 for data models.
- Format and lint with `ruff`. Run it before committing.
- Prefer small, single-purpose modules and functions. Avoid cleverness.
- Configuration goes in config files and environment variables. No hardcoded paths, URLs, model names, or credentials.
- Use structured logging. No stray `print` statements in library code.
- Handle errors explicitly. Do not swallow exceptions silently.

## Testing

- Use `pytest`. New behavior needs tests in the matching `tests/` directory.
- Mock all network and LLM calls in unit tests. Tests must pass offline.
- Run the tests for the half you changed, plus the contract tests if you touched `contracts/`.
- Do not delete or weaken a failing test to make a change pass. Fix the cause or flag it.

## Secrets and data

- Never commit `.env` files, API keys, tokens, or credentials. Provide a `.env.example` with placeholder names only.
- Never log secrets or full prompts containing user data.
- Large artifacts belong outside git. Reference them by path or download script.

## Git and PR workflow

- Branch names: `ai/<short-topic>`, `ml/<short-topic>`, or `contracts/<short-topic>`.
- Commit messages use conventional commits, for example `feat(ai): add hybrid retrieval` or `fix(ml): correct novelty time filter`.
- Keep PRs small and scoped to one half. A PR touching both `ai/` and `ml/` must be limited to `contracts/` changes and their direct consequences.
- Every PR description states what changed, how it was tested, and whether `contracts/` changed.
- Update the relevant section of `README.md` when behavior, interfaces, or setup change.

## How agents should work

- Explore before editing. Read the related files and existing patterns, then follow them.
- Make the smallest change that solves the task. Do not refactor unrelated code.
- If the task is ambiguous or crosses a boundary, ask one clear question instead of guessing.
- Do not add dependencies without a reason. State the reason when you do.
- After finishing, summarize what you changed, what you tested, and anything left undone.
- Never claim something works without running it. If you could not run it, say so.