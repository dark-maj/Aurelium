from __future__ import annotations

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar
from pydantic import BaseModel, ValidationError

try:
    import openai
    from openai import AsyncOpenAI
    RateLimitError = openai.RateLimitError
    APITimeoutError = openai.APITimeoutError
    APIConnectionError = openai.APIConnectionError
    InternalServerError = openai.InternalServerError
except ImportError:
    openai = None
    AsyncOpenAI = None  # type: ignore

    class RateLimitError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class InternalServerError(Exception):
        pass

from ai.config import settings

logger = logging.getLogger("aurelium.ai.llm")

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base class for swappable LLM clients."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Returns plain text completion."""
        ...

    @abstractmethod
    async def complete_json(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        """Returns a validated Pydantic model instance from structured output."""
        ...


class GrokClient(BaseLLMClient):
    """
    Async Grok client interacting with the xAI API via the OpenAI SDK.
    Supports timeouts, exponential backoff with jitter, rate limit handling,
    structured output parsing, and token usage logging.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        client: AsyncOpenAI | None = None,
    ):
        self.api_key = api_key or settings.xai_api_key
        self.base_url = base_url or settings.xai_base_url
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries or settings.llm_max_retries

        if client is not None:
            self._client = client
        else:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,  # We manage our own explicit retries with jitter and logging
            )

    async def _execute_with_retry(self, api_coroutine_fn: Any) -> Any:
        """Executes an API call with exponential backoff and jitter."""
        attempt = 0
        base_delay = 1.0

        while True:
            attempt += 1
            try:
                return await api_coroutine_fn()
            except (
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
            ) as exc:
                if attempt > self.max_retries:
                    logger.error(
                        "LLM call exceeded max retries",
                        extra={"attempt": attempt, "model": self.model, "error": str(exc)},
                    )
                    raise

                jitter = random.uniform(0.0, 0.5 * base_delay)
                sleep_duration = base_delay + jitter
                logger.warning(
                    "LLM call failed, retrying with backoff",
                    extra={
                        "attempt": attempt,
                        "sleep_s": round(sleep_duration, 2),
                        "model": self.model,
                        "error_type": exc.__class__.__name__,
                    },
                )
                await asyncio.sleep(sleep_duration)
                base_delay *= 2.0

    def _log_token_usage(self, response: Any, operation: str) -> None:
        """Logs token usage metrics in structured format."""
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            logger.info(
                "Grok API call succeeded",
                extra={
                    "operation": operation,
                    "model": self.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )

        response = await self._execute_with_retry(_call)
        self._log_token_usage(response, operation="complete")
        return response.choices[0].message.content or ""

    async def complete_json(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        """
        Attempts structured output via OpenAI/xAI beta.chat.completions.parse.
        Falls back to JSON mode validation with a single correction retry on error.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 1. Attempt native structured outputs (beta parse)
        try:
            async def _parse_call():
                return await self._client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=schema_model,
                    temperature=temperature,
                )

            response = await self._execute_with_retry(_parse_call)
            self._log_token_usage(response, operation="complete_json_parse")
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception as parse_err:
            logger.debug(
                "Native structured output parse failed; falling back to JSON schema mode",
                extra={"reason": str(parse_err), "model": self.model},
            )

        # 2. Fallback: JSON mode with Pydantic validation and 1 correction retry
        schema_json = json.dumps(schema_model.model_json_schema(), indent=2)
        json_sys_prompt = (
            (system_prompt + "\n\n" if system_prompt else "")
            + f"You must respond with valid JSON matching this schema:\n{schema_json}"
        )

        fallback_messages = [
            {"role": "system", "content": json_sys_prompt},
            {"role": "user", "content": prompt},
        ]

        async def _json_call(msgs):
            return await self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                response_format={"type": "json_object"},
                temperature=temperature,
            )

        response = await self._execute_with_retry(lambda: _json_call(fallback_messages))
        self._log_token_usage(response, operation="complete_json_fallback")
        raw_content = response.choices[0].message.content or "{}"

        try:
            return schema_model.model_validate_json(raw_content)
        except (ValidationError, json.JSONDecodeError) as val_err:
            logger.warning(
                "JSON fallback validation failed; retrying once with error feedback",
                extra={"error": str(val_err)},
            )
            # Correction retry with validation error feedback appended
            retry_messages = list(fallback_messages)
            retry_messages.append({"role": "assistant", "content": raw_content})
            retry_messages.append({
                "role": "user",
                "content": f"The previous output failed validation with error:\n{val_err}\nPlease fix the JSON according to the schema.",
            })

            retry_response = await self._execute_with_retry(lambda: _json_call(retry_messages))
            self._log_token_usage(retry_response, operation="complete_json_retry")
            corrected_content = retry_response.choices[0].message.content or "{}"
            return schema_model.model_validate_json(corrected_content)


def get_llm_client() -> BaseLLMClient:
    """Factory dependency for the configured LLM client."""
    return GrokClient()
