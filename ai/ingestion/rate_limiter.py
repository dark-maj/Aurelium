from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable, Coroutine
import httpx

logger = logging.getLogger("aurelium.ai.ingestion.rate_limiter")


class AsyncRateLimiter:
    """Token-bucket or sleep-based rate limiter for asynchronous HTTP requests."""

    def __init__(self, requests_per_second: float = 1.0):
        self.interval = 1.0 / max(0.1, requests_per_second)
        self.lock = asyncio.Lock()
        self.last_request_time = 0.0

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                wait_time = self.interval - elapsed
                await asyncio.sleep(wait_time)
            self.last_request_time = time.monotonic()


async def execute_with_backoff(
    request_coro_fn: Callable[[], Coroutine[Any, Any, httpx.Response]],
    max_retries: int = 4,
    base_delay: float = 1.0,
    service_name: str = "HTTP",
) -> httpx.Response:
    """
    Executes an HTTP request with exponential backoff and randomized jitter
    when encountering HTTP 429 (rate limit), 503, 502, or network timeouts.
    """
    attempt = 0
    current_delay = base_delay

    while True:
        attempt += 1
        try:
            resp = await request_coro_fn()
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt > max_retries:
                    logger.error(
                        f"{service_name} request failed after max retries",
                        extra={"status": resp.status_code, "attempt": attempt},
                    )
                    resp.raise_for_status()

                # Honor Retry-After header if returned
                retry_after_header = resp.headers.get("Retry-After")
                if retry_after_header and retry_after_header.isdigit():
                    sleep_time = float(retry_after_header)
                else:
                    jitter = random.uniform(0.0, 0.5 * current_delay)
                    sleep_time = current_delay + jitter
                    current_delay *= 2.0

                logger.warning(
                    f"{service_name} rate limit or server error ({resp.status_code}); backing off",
                    extra={"attempt": attempt, "sleep_time": round(sleep_time, 2)},
                )
                await asyncio.sleep(sleep_time)
                continue

            resp.raise_for_status()
            return resp

        except (httpx.TimeoutException, httpx.NetworkError) as net_err:
            if attempt > max_retries:
                logger.error(
                    f"{service_name} network failure after max retries",
                    extra={"error": str(net_err), "attempt": attempt},
                )
                raise

            jitter = random.uniform(0.0, 0.5 * current_delay)
            sleep_time = current_delay + jitter
            current_delay *= 2.0
            logger.warning(
                f"{service_name} connection error; retrying",
                extra={"attempt": attempt, "error": str(net_err), "sleep_time": round(sleep_time, 2)},
            )
            await asyncio.sleep(sleep_time)
