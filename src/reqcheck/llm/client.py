"""OpenAI client wrapper for LLM-powered analysis."""

import json
import logging
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
from reqcheck.llm.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Error during LLM API call."""

    pass


class LLMClient:
    """Wrapper for OpenAI API with retry logic and response parsing."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client: OpenAI | None = None

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
                raise LLMClientError(
                    "OpenAI API key not configured. "
                    "Set REQCHECK_OPENAI_API_KEY environment variable."
                )
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
        """Make API call with retry logic and exponential backoff.

        Args:
            user_prompt: The prompt to send to the API.

        Returns:
            The API response content.

        Raises:
            LLMClientError: If all retry attempts fail or a non-retryable error occurs.
        """
        max_retries = self._settings.llm_max_retries
        last_error: OpenAIError | None = None

        for attempt in range(max_retries + 1):
            try:
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
                    raise LLMClientError("Empty response from API")
                return content

            except OpenAIError as e:
                last_error = e

                if not self._is_retryable_error(e):
                    logger.error(f"OpenAI API error (non-retryable): {e}")
                    raise LLMClientError(f"API call failed: {e}") from e

                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"OpenAI API error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"OpenAI API error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"No more retries."
                    )

        # All retries exhausted
        raise LLMClientError(
            f"API call failed after {max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response with error handling."""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            raise LLMClientError(f"Invalid JSON response: {e}") from e

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
