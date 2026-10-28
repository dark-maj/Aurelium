from functools import lru_cache
from config import settings
from retrieval.vector_store import InMemoryVectorStore
from retrieval.retriever import HybridRetriever
from ml_interfaces.protocols import (
    RankerProtocol,
    NoveltyProtocol,
    TrendProtocol,
    ImpactProtocol,
)
from ml_interfaces.stubs import (
    StubRanker,
    StubNoveltyScorer,
    StubTrendForecaster,
    StubImpactPredictor,
)
from ml_interfaces.client import MLServiceClient


@lru_cache()
def get_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@lru_cache()
def get_ranker() -> RankerProtocol:
    if settings.use_ml_stubs:
        return StubRanker()
    return MLServiceClient(base_url=settings.ml_service_url)


@lru_cache()
def get_novelty_scorer() -> NoveltyProtocol:
    if settings.use_ml_stubs:
        return StubNoveltyScorer()
    return MLServiceClient(base_url=settings.ml_service_url)


@lru_cache()
def get_trend_forecaster() -> TrendProtocol:
    if settings.use_ml_stubs:
        return StubTrendForecaster()
    return MLServiceClient(base_url=settings.ml_service_url)


@lru_cache()
def get_impact_predictor() -> ImpactProtocol:
    if settings.use_ml_stubs:
        return StubImpactPredictor()
    return MLServiceClient(base_url=settings.ml_service_url)


def get_retriever() -> HybridRetriever:
    return HybridRetriever(
        vector_store=get_vector_store(),
        ranker=get_ranker(),
        novelty_scorer=get_novelty_scorer(),
        impact_predictor=get_impact_predictor(),
    )
