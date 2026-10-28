from __future__ import annotations

from typing import Sequence, Optional
import httpx

from schemas.models import Paper
from ml_interfaces.protocols import (
    RankerProtocol,
    NoveltyProtocol,
    TrendProtocol,
    ImpactProtocol,
)
from ml_interfaces.types import TrendSeries, TrendPoint


class MLServiceClient(RankerProtocol, NoveltyProtocol, TrendProtocol, ImpactProtocol):
    """
    HTTP Client implementation of all four ML protocols.
    Communicates with the teammate's ML microservice conforming to ml_service_openapi.yaml.
    """

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def rank(self, query: str, papers: Sequence[Paper]) -> list[float]:
        payload = {
            "query": query,
            "papers": [
                {
                    "id": p.id,
                    "title": p.title,
                    "abstract": p.abstract,
                    "embedding": p.embedding,
                }
                for p in papers
            ],
        }
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post("/ml/v1/ranking", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["scores"]

    def score(self, paper: Paper) -> float:
        payload = {
            "id": paper.id,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "published_date": paper.published_date.isoformat() if paper.published_date else None,
            "embedding": paper.embedding,
        }
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post("/ml/v1/novelty", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return float(data["novelty_score"])

    def forecast(self, topic: str) -> TrendSeries:
        payload = {"topic": topic}
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post("/ml/v1/trends/forecast", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return TrendSeries(
                topic=data["topic"],
                historical_points=[TrendPoint(**p) for p in data.get("historical_points", [])],
                forecast_points=[TrendPoint(**p) for p in data.get("forecast_points", [])],
                growth_rate=data.get("growth_rate", 0.0),
                is_emerging=data.get("is_emerging", False),
            )

    def predict(self, paper: Paper) -> float:
        payload = {
            "id": paper.id,
            "authors_count": len(paper.authors),
            "venue": paper.venue,
            "fields_of_study": paper.fields_of_study,
            "citation_count": paper.citation_count,
            "embedding": paper.embedding,
        }
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post("/ml/v1/impact", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return float(data["predicted_citations"])
