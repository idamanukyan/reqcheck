"""OpenAI client wrapper for LLM-powered analysis."""

import json
import logging
from typing import Any

from openai import OpenAI, OpenAIError

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

    @property
    def client(self) -> OpenAI:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            if not self._settings.openai_api_key:
                raise LLMClientError(
                    "OpenAI API key not configured. Set QA_AGENT_OPENAI_API_KEY environment variable."
                )
            self._client = OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def _call_api(self, user_prompt: str) -> str:
        """Make API call with error handling."""
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
            logger.error(f"OpenAI API error: {e}")
            raise LLMClientError(f"API call failed: {e}") from e

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
