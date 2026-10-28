from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, TypeVar
from pydantic import BaseModel

logger = logging.getLogger("aurelium.ai.agents.cache")

T = TypeVar("T", bound=BaseModel)


class LLMCache:
    """
    Deterministic cache for LLM results keyed by:
    prompt_version : model : sha256(input_payload).
    Prevents redundant LLM API calls and cost on reruns.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._store: dict[str, str] = {}

    def _make_key(self, prompt_version: str, model: str, input_payload: str) -> str:
        payload_hash = hashlib.sha256(input_payload.encode("utf-8")).hexdigest()
        return f"{prompt_version}:{model}:{payload_hash}"

    def get(
        self,
        prompt_version: str,
        model: str,
        input_payload: str,
        schema_model: type[T],
    ) -> T | None:
        """Retrieves cached validated Pydantic model if present."""
        if not self.enabled:
            return None

        key = self._make_key(prompt_version, model, input_payload)
        cached_json = self._store.get(key)
        if cached_json is not None:
            logger.debug(f"Cache HIT for LLM call: {key[:32]}...")
            return schema_model.model_validate_json(cached_json)

        return None

    def get_str(
        self,
        prompt_version: str,
        model: str,
        input_payload: str,
    ) -> str | None:
        """Retrieves cached string completion if present."""
        if not self.enabled:
            return None
        key = self._make_key(prompt_version, model, input_payload)
        return self._store.get(key)

    def set(
        self,
        prompt_version: str,
        model: str,
        input_payload: str,
        value: BaseModel | str,
    ) -> None:
        """Stores result in cache."""
        if not self.enabled:
            return

        key = self._make_key(prompt_version, model, input_payload)
        if isinstance(value, BaseModel):
            self._store[key] = value.model_dump_json()
        else:
            self._store[key] = str(value)

    def clear(self) -> None:
        """Flushes the cache."""
        self._store.clear()

    def size(self) -> int:
        return len(self._store)


# Global default cache
global_llm_cache = LLMCache()
