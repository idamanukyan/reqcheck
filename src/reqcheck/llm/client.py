"""Provider-agnostic LLM client wrapper for analysis.

This client works with any LLM provider (OpenAI, Anthropic, etc.) through
the provider abstraction layer in providers.py.
"""

import json
import random
import time
from typing import Any

from reqcheck.core.config import Settings, get_settings
from reqcheck.core.exceptions import (
    LLMCircuitBreakerError,
    LLMClientError,
    LLMRateLimitError,
    LLMResponseError,
    ReqcheckError,
)
from reqcheck.core.logging import get_logger
from reqcheck.core.observability import MetricNames, get_metrics
from reqcheck.llm.circuit_breaker import (
    CircuitBreakerConfig,
    get_circuit_breaker,
)
from reqcheck.llm.prompts import PromptTemplates
from reqcheck.llm.providers import BaseProvider, LLMResponse, create_provider

logger = get_logger("llm.client")


class LLMClient:
    """Provider-agnostic LLM client with retry logic, circuit breaker, and response parsing.

    This client can work with any provider (OpenAI, Anthropic, etc.) through the
    provider abstraction layer. The provider is determined by the llm_provider
    setting.

    Usage:
        # Default: uses provider from settings
        client = LLMClient()

        # With custom settings
        client = LLMClient(settings=custom_settings)

        # With explicit provider
        from reqcheck.llm.providers import OpenAIProvider, ProviderConfig
        provider = OpenAIProvider(ProviderConfig(api_key="...", model="gpt-4"))
        client = LLMClient(provider=provider)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        provider: BaseProvider | None = None,
    ):
        self._settings = settings or get_settings()
        self._provider = provider
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
    def provider(self) -> BaseProvider:
        """Lazy-initialize the LLM provider."""
        if self._provider is None:
            self._provider = create_provider(self._settings)
        return self._provider

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider."""
        return self.provider.name

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

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable.

        Handles errors from any provider by checking error type names
        and common patterns.

        Args:
            error: The exception to check.

        Returns:
            True if the error is retryable, False otherwise.
        """
        error_type = type(error).__name__.lower()

        # Common retryable error patterns across providers
        retryable_patterns = [
            "timeout",
            "connection",
            "ratelimit",
            "rate_limit",
            "serverError",
            "internalserver",
            "overloaded",
            "serviceunavailable",
        ]

        return any(pattern in error_type for pattern in retryable_patterns)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit error."""
        error_type = type(error).__name__.lower()
        return "ratelimit" in error_type or "rate_limit" in error_type

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
        provider_name = self.provider_name

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
                        "provider": provider_name,
                    },
                )
                raise LLMCircuitBreakerError(retry_after)

        max_retries = self._settings.llm_max_retries
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                metrics.increment(MetricNames.LLM_CALLS, tags={"provider": provider_name})

                with metrics.timer(MetricNames.LLM_LATENCY, tags={"provider": provider_name}):
                    response: LLMResponse = self.provider.complete(
                        system_prompt=PromptTemplates.SYSTEM,
                        user_prompt=user_prompt,
                        response_format="json",
                    )

                content = response.content
                if not content:
                    raise LLMResponseError("Empty response from API")

                # Record success with circuit breaker
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()

                # Record token usage
                if response.usage:
                    metrics.increment(
                        MetricNames.LLM_TOKENS_USED,
                        value=response.usage.total_tokens,
                        tags={"provider": provider_name},
                    )

                return content

            except ReqcheckError:
                # Re-raise our own errors (like LLMResponseError)
                raise
            except Exception as e:
                last_error = e
                metrics.increment(
                    MetricNames.LLM_ERRORS,
                    tags={"provider": provider_name, "type": type(e).__name__},
                )

                if not self._is_retryable_error(e):
                    # Record failure with circuit breaker
                    if self._circuit_breaker is not None:
                        self._circuit_breaker.record_failure()

                    logger.error(
                        "LLM API error (non-retryable)",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "provider": provider_name,
                            "retryable": False,
                        },
                    )
                    raise LLMClientError(
                        f"API call failed: {e}",
                        provider=provider_name,
                        retryable=False,
                    ) from e

                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        "LLM API error, retrying",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": max_retries + 1,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "provider": provider_name,
                            "retry_delay_seconds": round(delay, 2),
                        },
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "LLM API error, no more retries",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": max_retries + 1,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "provider": provider_name,
                        },
                    )

        # Record failure with circuit breaker (all retries exhausted)
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_failure()

        # All retries exhausted - check if it was a rate limit
        if last_error and self._is_rate_limit_error(last_error):
            raise LLMRateLimitError() from last_error

        raise LLMClientError(
            f"API call failed after {max_retries + 1} attempts: {last_error}",
            provider=provider_name,
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
        return self.provider.is_available()

    def circuit_breaker_stats(self) -> dict[str, Any] | None:
        """Get circuit breaker statistics, or None if disabled."""
        if self._circuit_breaker is None:
            return None
        return self._circuit_breaker.stats.to_dict()
