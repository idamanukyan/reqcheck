"""Async OpenAI client wrapper for LLM-powered analysis."""

import asyncio
import json
import random
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)

from reqcheck.core.config import Settings, get_settings
from reqcheck.core.exceptions import (
    LLMCircuitBreakerError,
    LLMClientError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
)
from reqcheck.core.logging import get_logger
from reqcheck.core.observability import MetricNames, get_metrics
from reqcheck.llm.cache import LLMCache, get_cache
from reqcheck.llm.circuit_breaker import (
    CircuitBreakerConfig,
    get_circuit_breaker,
)
from reqcheck.llm.prompts import PromptTemplates
from reqcheck.llm.providers import TokenUsage
from reqcheck.llm.usage import get_usage_tracker

logger = get_logger("llm.async_client")


class AsyncLLMClient:
    """Async wrapper for OpenAI API with circuit breaker, retry logic, caching, and response parsing."""

    # Error types that should trigger a retry
    RETRYABLE_ERRORS = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )

    def __init__(
        self,
        settings: Settings | None = None,
        cache: LLMCache | None = None,
    ):
        self._settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
        self._cache = cache or get_cache()
        self._circuit_breaker = self._init_circuit_breaker()

    def _init_circuit_breaker(self):
        """Initialize circuit breaker if enabled."""
        if not self._settings.circuit_breaker_enabled:
            return None

        config = CircuitBreakerConfig(
            failure_threshold=self._settings.circuit_breaker_failure_threshold,
            recovery_timeout=self._settings.circuit_breaker_recovery_timeout,
            success_threshold=self._settings.circuit_breaker_success_threshold,
            failure_window=self._settings.circuit_breaker_failure_window,
        )
        return get_circuit_breaker(config)

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize AsyncOpenAI client."""
        if self._client is None:
            if not self._settings.openai_api_key:
                raise LLMConfigurationError("OPENAI_API_KEY")
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter.

        Args:
            attempt: The current attempt number (0-indexed).

        Returns:
            Delay in seconds before the next retry.
        """
        base_delay = self._settings.llm_retry_base_delay
        max_delay = self._settings.llm_retry_max_delay

        # Exponential backoff: base_delay * 2^attempt
        delay = base_delay * (2**attempt)

        # Add jitter (±25% randomization) to prevent thundering herd
        jitter = delay * 0.25 * (2 * random.random() - 1)
        delay += jitter

        # Cap at maximum delay
        return min(delay, max_delay)

    def _is_retryable_error(self, error: OpenAIError) -> bool:
        """Check if an error is retryable."""
        return isinstance(error, self.RETRYABLE_ERRORS)

    async def _call_api(self, user_prompt: str) -> str:
        """Make async API call with circuit breaker, retry logic, and exponential backoff.

        Args:
            user_prompt: The prompt to send to the API.

        Returns:
            The API response content.

        Raises:
            LLMCircuitBreakerError: If circuit breaker is open.
            LLMClientError: If all retry attempts fail or a non-retryable error occurs.
        """
        metrics = get_metrics()

        # Check circuit breaker first
        if self._circuit_breaker is not None:
            if not self._circuit_breaker.can_execute():
                retry_after = self._circuit_breaker.time_until_retry()
                self._circuit_breaker.record_rejected()
                metrics.increment(MetricNames.CB_REJECTIONS)
                logger.warning(
                    "Circuit breaker rejecting request",
                    extra={
                        "retry_after": round(retry_after, 2),
                        "state": self._circuit_breaker.state.value,
                    },
                )
                raise LLMCircuitBreakerError(retry_after)

        max_retries = self._settings.llm_max_retries
        last_error: OpenAIError | None = None

        for attempt in range(max_retries + 1):
            try:
                metrics.increment(MetricNames.LLM_CALLS, tags={"provider": "openai"})
                start_time = asyncio.get_event_loop().time()

                response = await self.client.chat.completions.create(
                    model=self._settings.openai_model,
                    messages=[
                        {"role": "system", "content": PromptTemplates.SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._settings.openai_temperature,
                    max_tokens=self._settings.openai_max_tokens,
                    response_format={"type": "json_object"},
                )

                # Record latency
                elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                metrics.timing(MetricNames.LLM_LATENCY, elapsed_ms, tags={"provider": "openai"})

                content = response.choices[0].message.content
                if content is None:
                    raise LLMResponseError("Empty response from API")

                # Track token usage if enabled
                if self._settings.track_token_usage and response.usage:
                    usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                    get_usage_tracker().record_usage(
                        usage=usage,
                        model=self._settings.openai_model,
                        provider="openai",
                    )
                    metrics.increment(
                        MetricNames.LLM_TOKENS_USED,
                        value=response.usage.total_tokens,
                        tags={"provider": "openai"},
                    )

                # Record success with circuit breaker
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()

                return content

            except OpenAIError as e:
                last_error = e
                metrics.increment(
                    MetricNames.LLM_ERRORS,
                    tags={"provider": "openai", "type": type(e).__name__},
                )

                if not self._is_retryable_error(e):
                    # Record failure with circuit breaker
                    if self._circuit_breaker is not None:
                        self._circuit_breaker.record_failure()

                    logger.error(
                        "OpenAI API error (non-retryable)",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "retryable": False,
                        },
                    )
                    raise LLMClientError(
                        f"API call failed: {e}",
                        provider="openai",
                        retryable=False,
                    ) from e

                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        "OpenAI API error, retrying",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": max_retries + 1,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "retry_delay_seconds": round(delay, 2),
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "OpenAI API error, no more retries",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": max_retries + 1,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

        # Record failure with circuit breaker (all retries exhausted)
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_failure()

        # All retries exhausted - check if it was a rate limit
        if isinstance(last_error, RateLimitError):
            raise LLMRateLimitError() from last_error

        raise LLMClientError(
            f"API call failed after {max_retries + 1} attempts: {last_error}",
            provider="openai",
            retryable=True,
        ) from last_error

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response with error handling."""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON response",
                extra={
                    "error": str(e),
                    "response_length": len(response),
                    "response_preview": response[:200] if response else "(empty)",
                },
            )
            raise LLMResponseError(f"Invalid JSON response: {e}", raw_response=response) from e

    async def _cached_call(
        self, prompt_type: str, text: str, prompt: str
    ) -> dict[str, Any]:
        """Make a cached API call.

        Checks cache first, calls API if miss, caches successful responses.
        """
        metrics = get_metrics()

        # Check cache first
        cached = self._cache.get(prompt_type, text)
        if cached is not None:
            metrics.increment(MetricNames.LLM_CACHE_HITS, tags={"prompt_type": prompt_type})
            logger.debug(
                "Cache hit",
                extra={"prompt_type": prompt_type, "text_length": len(text)},
            )
            return cached

        # Cache miss - call API
        metrics.increment(MetricNames.LLM_CACHE_MISSES, tags={"prompt_type": prompt_type})
        logger.debug(
            "Cache miss, calling API",
            extra={"prompt_type": prompt_type, "text_length": len(text)},
        )
        response = await self._call_api(prompt)
        result = self._parse_json_response(response)

        # Cache the result
        self._cache.set(prompt_type, text, result)

        return result

    async def analyze_ambiguity(self, requirement_text: str) -> dict[str, Any]:
        """Analyze requirement for ambiguity issues."""
        prompt = PromptTemplates.format_ambiguity(requirement_text)
        return await self._cached_call("ambiguity", requirement_text, prompt)

    async def analyze_completeness(self, requirement_text: str) -> dict[str, Any]:
        """Analyze requirement for completeness issues."""
        prompt = PromptTemplates.format_completeness(requirement_text)
        return await self._cached_call("completeness", requirement_text, prompt)

    async def analyze_testability(self, requirement_text: str) -> dict[str, Any]:
        """Analyze requirement for testability issues."""
        prompt = PromptTemplates.format_testability(requirement_text)
        return await self._cached_call("testability", requirement_text, prompt)

    async def analyze_risk(self, requirement_text: str) -> dict[str, Any]:
        """Analyze requirement for risk signals."""
        prompt = PromptTemplates.format_risk(requirement_text)
        return await self._cached_call("risk", requirement_text, prompt)

    async def generate_summary(
        self,
        title: str,
        issues_summary: str,
        ambiguity_score: float,
        completeness_score: float,
        testability_score: float,
    ) -> dict[str, Any]:
        """Generate executive summary of analysis.

        Note: Summary is not cached as it depends on computed scores.
        """
        prompt = PromptTemplates.format_summary(
            title=title,
            issues_summary=issues_summary,
            ambiguity_score=ambiguity_score,
            completeness_score=completeness_score,
            testability_score=testability_score,
        )
        response = await self._call_api(prompt)
        return self._parse_json_response(response)

    def is_available(self) -> bool:
        """Check if LLM client is properly configured."""
        return bool(self._settings.openai_api_key)

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()

    def usage_stats(self) -> dict[str, Any]:
        """Get token usage statistics."""
        tracker = get_usage_tracker()
        return {
            "session": tracker.get_session_stats().to_dict(),
            "total": tracker.get_total_stats().to_dict(),
            "by_model": tracker.get_model_stats(),
        }

    def circuit_breaker_stats(self) -> dict[str, Any] | None:
        """Get circuit breaker statistics, or None if disabled."""
        if self._circuit_breaker is None:
            return None
        return self._circuit_breaker.stats.to_dict()
