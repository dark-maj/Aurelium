# Aurelium Shared Contracts

Shared contract package between the `ai/` and `ml/` halves of Aurelium.

## Structure

- `contracts/schemas/`: Pydantic v2 domain schemas (`Paper`, `Chunk`, `Summary`, `ComparisonTable`, `ComparisonCell`, `GapReport`, `Hypothesis`, `NoveltyScore`, `TrendSeries`, `ResearchResult`).
- `contracts/interfaces/`: `typing.Protocol` interfaces (`RankerProtocol`, `NoveltyProtocol`, `ImpactProtocol`, `TrendProtocol`) and deterministic stubs.
- `contracts/openapi/`: OpenAPI 3.0.3 specification for the ML microservice endpoints.

## Installation

```bash
# In editable mode from repo root
pip install -e ./contracts
```

## Usage

```python
from contracts.schemas import Paper, Chunk, ResearchResult
from contracts.interfaces import (
    RankerProtocol,
    NoveltyProtocol,
    ImpactProtocol,
    TrendProtocol,
    StubRanker,
    StubNoveltyScorer,
    StubImpactPredictor,
    StubTrendForecaster,
)
```
