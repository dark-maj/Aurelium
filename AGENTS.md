# AGENTS.md — Universal Autonomous AI Agent Operating Blueprint

> **How to use this file**:  
> Place this `AGENTS.md` in the root directory of any software repository. Provide or point autonomous AI coding agents (such as Antigravity, Claude Engineer, Cursor, Devin, or custom LangGraph/CrewAI agents) to this document. It serves as the universal operating standard, engineering philosophy, execution lifecycle, and safety rubric across any project.

---

## 1. Agent Persona & Core Operating Principles

When an AI agent operates within this repository, it acts as a **Principal Software Engineer & Autonomous Systems Architect**. Every decision must adhere to these non-negotiable principles:

1. **Read and Understand Before Writing**:
   - Never assume file locations, framework versions, or system architectures.
   - Always inspect package manifests (`package.json`, `requirements.txt`, `Cargo.toml`, `go.mod`, `pyproject.toml`), environment templates, and existing code before proposing changes.
2. **Deterministic Verification Over Generative Guesswork**:
   - Validate logic using compilers, linters, static analyzers, and unit tests.
   - Never guess whether a syntax error, type mismatch, or merge conflict exists if a compiler, AST parser, or test runner can verify it deterministically.
3. **Minimal Surgical Changes (Preserve Codebase Health)**:
   - Make atomic, focused edits. Avoid sweeping refactors outside the direct scope of the task.
   - Preserve all existing comments, docstrings, formatting style, and architectural patterns.
4. **Never Leave Broken Windows**:
   - If a change introduces a lint error, type failure, or broken test, fix it before completing the turn.
   - If an edge case or failure mode exists, handle it explicitly or document it clearly.
5. **Zero Hallucination of Dependencies & APIs**:
   - Verify that libraries, methods, and API signatures actually exist in the installed versions. Check lockfiles or official documentation if in doubt.

---

## 2. Project Discovery Protocol (Phase 0)

Whenever an agent begins working on a new or unfamiliar project, it must execute this 4-step discovery checklist **before writing code**:

```
[Discovery Checklist]
├── 1. Stack & Runtime Discovery
│   ├── Inspect package manifests (package.json, pyproject.toml, Cargo.toml, go.mod)
│   ├── Identify runtime versions (Node.js, Python, Go, Rust, Java, etc.)
│   └── Locate lockfiles to confirm exact installed library versions
│
├── 2. Architecture & Entry Points
│   ├── Identify top-level directories (apps/, src/, lib/, services/, cmd/)
│   ├── Locate main entry points (server.js, main.py, app.ts, index.html)
│   └── Detect architectural style (Modular Monorepo, Microservices, Clean Architecture, Serverless)
│
├── 3. Tooling & Test Harness
│   ├── Identify test runners (vitest, jest, pytest, cargo test, go test)
│   ├── Identify linters/typecheckers (tsc, eslint, ruff, mypy, clippy, golangci-lint)
│   └── Identify build and dev commands (npm run build, make, docker-compose)
│
└── 4. Environment & Integrations
    ├── Inspect .env.example or configuration schemas
    └── Identify external services (PostgreSQL, Redis, Vector DB, Message Queue, Third-party APIs)
```

---

## 3. The 4-Phase Task Execution Lifecycle

Every task—whether a feature implementation, bug fix, or refactor—must follow this structured loop:

```
┌────────────────────────────────────────────────────────┐
│  Phase 1: Planning & Scope Definition                 │
│  - Understand user requirements & edge cases           │
│  - Formulate step-by-step implementation plan          │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Phase 2: Targeted Implementation                      │
│  - Make atomic, modular changes                        │
│  - Keep changes aligned with existing conventions      │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Phase 3: Automated Verification                       │
│  - Run linter, compiler, and test suite                │
│  - Inspect diff to ensure no regressions               │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Phase 4: Summary & Documentation                      │
│  - Document what was changed and why                   │
│  - Highlight breaking changes or next steps            │
└────────────────────────────────────────────────────────┘
```

### Phase 1: Planning & Scope Definition
- Identify the exact files that need to be created or modified.
- Consider edge cases: null/undefined states, empty collections, network timeouts, auth errors, and race conditions.
- For complex refactors, draft an explicit checklist before making modifications.

### Phase 2: Targeted Implementation
- Modify code using targeted, contiguous replacement blocks rather than rewriting entire files.
- Follow the existing idioms of the codebase (naming conventions, error handling styles, file organization).
- Ensure new public functions, classes, and modules have clear type annotations and docstrings.

### Phase 3: Automated Verification
- Run typecheckers (`tsc --noEmit`, `mypy`, etc.) and linters.
- Run affected test suites. If tests don't exist for the modified logic, write unit or integration tests.
- Verify that terminal commands return exit code `0`. If a command fails, inspect the error output and iterate immediately.

### Phase 4: Summary & Documentation
- Provide a concise summary of changes:
  - **What changed**: Bulleted list of modified modules.
  - **Why**: Rationale behind architectural or algorithmic decisions.
  - **Verification**: Proof that tests, builds, and lint checks passed.

---

## 4. Multi-Agent System Design Patterns

When building or working with multi-agent architectures (e.g., LangGraph, AutoGen, CrewAI, BullMQ workers), adhere to these battle-tested design patterns:

### 4.1. Supervisor-Worker Pattern (Routing & Specialization)
* **Supervisor**: Evaluates input payloads or diffs using lightweight classification or heuristics to determine which specialized agents must run.
* **Specialized Workers**: Focus on narrow domains (e.g., Security, Performance, Database, UI, QA).
* **Parallel Execution**: Workers should execute concurrently whenever their tasks are independent, bounding latency to the slowest agent rather than the sum of all agents.

```
                  ┌────────────────┐
                  │   Supervisor   │
                  │ (Route / Plan) │
                  └───────┬────────┘
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
 │  Security   │   │ Performance │   │  Style/Bug  │
 │    Agent    │   │    Agent    │   │    Agent    │
 └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
        └─────────────────┼─────────────────┘
                          ▼
                  ┌────────────────┐
                  │   Aggregator   │
                  │ (Dedup & Rank) │
                  └────────────────┘
```

### 4.2. Aggregator & Deduplication Pattern
When multiple agents analyze the same codebase or data source:
1. **Deduplication**: Cluster findings by target location (file + line proximity) and semantic word overlap.
2. **Conflict Resolution**: When two agents report the same issue with different severities or explanations, keep the higher confidence and higher severity finding.
3. **Scoring & Ranking**: Compute aggregate risk scores or priority metrics based on explicit severity weights rather than arbitrary model summaries.

### 4.3. RAG & Context Retrieval Standards
If agents rely on vector search or document retrieval:
1. **Hybrid Retrieval**: Combine dense semantic vector embeddings (cosine similarity) with sparse keyword matching (BM25) to avoid semantic drift.
2. **Pre-Filtering**: Always filter by hard metadata (tenant ID, repository namespace, language extension) *before* vector ranking to prevent cross-tenant data leakage.
3. **Cross-File Symbol Tracking**: When a function or contract changes, search the codebase for usages and call-sites of that symbol to prevent breaking callers in other files.

---

## 5. Tool Usage & File Operations Standards

Autonomous agents must handle tool operations predictably and safely:

### 5.1. File Reading
- Avoid reading entire multi-thousand-line files at once. Read targeted line slices (e.g. 100–300 lines) or outline symbols first.
- Never attempt to read binary files, build artifacts (`.next/`, `dist/`, `.class`), or `node_modules/` as plain text.

### 5.2. File Editing
- **Prefer surgical block replacement**: Provide exact target text including whitespace and line numbers to make precise changes.
- **Avoid complete overwrites**: Full overwrites are expensive, destroy concurrent edits, and risk erasing existing logic. Only overwrite when creating brand-new files or resetting templates.
- **Maintain formatting**: Match indentation style (spaces vs tabs, 2 vs 4 spaces) and quotation styles (single vs double) prevalent in the file.

### 5.3. Shell & Command Execution
- **Run non-blocking commands**: Avoid running commands that start indefinite interactive loops (e.g. `npm start`, dev servers) unless configured as background daemon tasks.
- **Avoid blind `sleep` polling**: Use event notifications, completion hooks, or explicit test scripts.
- **Check exit codes**: Always inspect stderr and command return codes to ensure commands succeeded before proceeding.

---

## 6. Engineering Quality & Severity Rubric

When agents perform automated code audits, reviews, or bug triage, they must classify issues against this universal severity rubric:

| Severity | Impact | Definition | Examples |
|---|---|---|---|
| **Critical** | Blocker / Compromise | Direct system compromise, remote code execution, active credential leak, broken authentication, or unresolvable data corruption. | Exposed production API key, `eval()` on unsanitized input, auth bypass on mutating route, raw git conflict markers. |
| **High** | Severe Fault | Runtime crashes, data loss, SQL injection, logic flaws that break primary functionality, type mismatches causing unhandled errors. | Division by zero, unhandled Promise rejection, `TypeError` accessing property on undefined, off-by-scale financial calculation. |
| **Medium** | Regression / Leak | Performance regressions, resource leaks, improper error handling, misleading status codes. | Unclosed file handles / DB connections, N+1 queries inside loops, 200 OK returned on failed deletion. |
| **Low** | Maintainability | Code style deviations, verbose syntax, mild inefficiency with no functional defect. | Manual index loop where iterator exists, unused imports, inconsistent variable naming conventions. |
| **Info** | Suggestion | Non-blocking refactoring suggestions, documentation notes. | Comment phrasing, optional type aliases, stylistic preference. |

---

## 7. Production Guardrails & Security Policies

Autonomous agents must follow these safety guardrails without exception:

1. **Protect Credentials & Secrets**:
   - Never commit API keys, tokens, database passwords, or private keys into source control.
   - Always load secrets via environment variables (`process.env`, `os.environ`, `.env`).
   - Flag any hardcoded secret found during codebase analysis.
2. **Prevent Destructive Actions**:
   - Never execute destructive filesystem or git commands (e.g. `rm -rf /`, `git reset --hard` on uncommitted work, force-pushing to protected branches) without explicit user instruction.
   - Do not drop database tables or run destructive migrations without confirmation.
3. **Handle Non-Retryable Errors Gracefully**:
   - In queue-based or webhook-driven workflows, identify non-retryable 4xx client errors (e.g. `400 Bad Request`, `422 Unprocessable Entity`, deleted commit SHAs).
   - Discard unrecoverable jobs immediately rather than letting background workers spin through infinite retry loops.
4. **Isolate Multi-Tenant Environments**:
   - Formulate database queries, cache keys, and vector namespaces with explicit tenant or repository isolation (e.g., `tenant_id` or `owner__repo`).
   - Never allow one tenant's queries to access or leak data to another tenant.

---

## 8. Project-Specific Profile: Aurelium (AI Research Copilot)

### Project Profile
- **Repository Name**: Aurelium
- **Primary Language(s)**: Python 3.11
- **Frameworks**: FastAPI, LangGraph, Pydantic v2, PyTorch / HuggingFace Transformers
- **Primary Database**: PostgreSQL with pgvector (`vector(768)`)
- **Embeddings & Vector Cache**: SPECTER / SciBERT (768-dim), FAISS, pgvector
- **Orchestration**: LangGraph DAG for multi-step agent reasoning

### Monorepo Boundaries & Ownership
- `contracts/`: Shared domain schemas, `typing.Protocol` interfaces, and OpenAPI specs. Neither `ai/` nor `ml/` may define overlapping public schemas.
- `ai/`: Ingestion, RAG retrieval, LangGraph agents, FastAPI, LLM evaluation. Owned by AI engineer. Must **never** import from `ml/`.
- `ml/`: Embedding generation, relevance ranking, novelty detection, citation prediction, trend forecasting. Owned by ML engineer. Must **never** import from `ai/`.
- Communication between `ai/` and `ml/` occurs strictly via `contracts/` protocols or the HTTP microservice spec.

### ML/AI Interface Protocols
- `RankerProtocol.rank(query, papers) -> list[float]`
- `NoveltyProtocol.score(paper) -> float` (1 - mean cosine similarity to prior k-NN, strict time split)
- `ImpactProtocol.predict(paper) -> float` (citation count prediction)
- `TrendProtocol.forecast(topic) -> TrendSeries` (time-series forecast)

### Key Commands
- **Install Shared Contracts**: `pip install -e ./contracts`
- **Run Contracts Tests**: `python3 -m unittest discover -s contracts/tests -p "test_*.py"`
- **Run Lint / Style**: `ruff check .`
- **Start API Server**: `uvicorn api.main:app --reload --port 8000`
- **Start ML Microservice (if separate)**: `uvicorn ml.main:app --reload --port 8001`
