"""OpenAI client wrapper for LLM-powered analysis."""

import json
import random
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
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
from reqcheck.llm.circuit_breaker import (
    CircuitBreakerConfig,
    get_circuit_breaker,
)
from reqcheck.llm.prompts import PromptTemplates

logger = get_logger("llm.client")


class LLMClient:
    """Wrapper for OpenAI API with retry logic, circuit breaker, and response parsing."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client: OpenAI | None = None
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

    # Error types that should trigger a retry
    RETRYABLE_ERRORS = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        InternalServerError,
    )

    @property
    def client(self) -> OpenAI:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            if not self._settings.openai_api_key:
                raise LLMConfigurationError("OPENAI_API_KEY")
            self._client = OpenAI(api_key=self._settings.openai_api_key)
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
        """Check if an error is retryable.

        Args:
            error: The OpenAI error to check.

        Returns:
            True if the error is retryable, False otherwise.
        """
        return isinstance(error, self.RETRYABLE_ERRORS)

    def _call_api(self, user_prompt: str) -> str:
        """Make API call with circuit breaker, retry logic, and exponential backoff.

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

                with metrics.timer(MetricNames.LLM_LATENCY, tags={"provider": "openai"}):
                    response = self.client.chat.completions.create(
                        model=self._settings.openai_model,
                        messages=[
                            {"role": "system", "content": PromptTemplates.SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self._settings.openai_temperature,
                        max_tokens=self._settings.openai_max_tokens,
                        response_format={"type": "json_object"},
                    )

                content = response.choices[0].message.content
                if content is None:
                    raise LLMResponseError("Empty response from API")

                # Record success with circuit breaker
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()

                # Record token usage
                if response.usage:
                    metrics.increment(
                        MetricNames.LLM_TOKENS_USED,
                        value=response.usage.total_tokens,
                        tags={"provider": "openai"},
                    )

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
                    time.sleep(delay)
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

    def analyze_ambiguity(self, requirement_text: str) -> dict[str, Any]:
        """Analyze requirement for ambiguity issues."""
        prompt = PromptTemplates.format_ambiguity(requirement_text)
        response = self._call_api(prompt)
        return self._parse_json_response(response)

    def analyze_completeness(self, requirement_text: str) -> dict[str, Any]:
        """Analyze requirement for completeness issues."""
        prompt = PromptTemplates.format_completeness(requirement_text)
        response = self._call_api(prompt)
        return self._parse_json_response(response)

    def analyze_testability(self, requirement_text: str) -> dict[str, Any]:
        """Analyze requirement for testability issues."""
        prompt = PromptTemplates.format_testability(requirement_text)
        response = self._call_api(prompt)
        return self._parse_json_response(response)

    def analyze_risk(self, requirement_text: str) -> dict[str, Any]:
        """Analyze requirement for risk signals."""
        prompt = PromptTemplates.format_risk(requirement_text)
        response = self._call_api(prompt)
        return self._parse_json_response(response)

    def generate_summary(
        self,
        title: str,
        issues_summary: str,
        ambiguity_score: float,
        completeness_score: float,
        testability_score: float,
    ) -> dict[str, Any]:
        """Generate executive summary of analysis."""
        prompt = PromptTemplates.format_summary(
            title=title,
            issues_summary=issues_summary,
            ambiguity_score=ambiguity_score,
            completeness_score=completeness_score,
            testability_score=testability_score,
        )
        response = self._call_api(prompt)
        return self._parse_json_response(response)

    def is_available(self) -> bool:
        """Check if LLM client is properly configured."""
        return bool(self._settings.openai_api_key)

    def circuit_breaker_stats(self) -> dict[str, Any] | None:
        """Get circuit breaker statistics, or None if disabled."""
        if self._circuit_breaker is None:
            return None
        return self._circuit_breaker.stats.to_dict()
