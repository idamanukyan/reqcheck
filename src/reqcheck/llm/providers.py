"""LLM provider abstraction layer.

This module provides a unified interface for different LLM providers,
allowing easy switching between OpenAI, Anthropic, and other providers.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from reqcheck.core.config import Settings
from reqcheck.core.exceptions import LLMConfigurationError, LLMResponseError
from reqcheck.core.logging import get_logger

logger = get_logger("llm.providers")


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: "TokenUsage"
    raw_response: Any = None


@dataclass
class TokenUsage:
    """Token usage tracking for cost estimation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Cached tokens (for providers that support it)
    cached_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    api_key: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 2000
    extra_params: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        ...

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Make a synchronous completion request.

        Args:
            system_prompt: The system message
            user_prompt: The user message
            response_format: Optional format hint ('json' for JSON mode)

        Returns:
            Standardized LLM response
        """
        ...

    async def complete_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Make an asynchronous completion request.

        Args:
            system_prompt: The system message
            user_prompt: The user message
            response_format: Optional format hint ('json' for JSON mode)

        Returns:
            Standardized LLM response
        """
        ...

    def is_available(self) -> bool:
        """Check if the provider is properly configured."""
        ...


class BaseProvider(ABC):
    """Base class for LLM providers with common functionality."""

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate provider configuration."""
        if not self._config.api_key:
            raise LLMConfigurationError(f"{self.name.upper()}_API_KEY")

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Make a synchronous completion request."""
        pass

    @abstractmethod
    async def complete_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Make an asynchronous completion request."""
        pass

    def is_available(self) -> bool:
        """Check if the provider is properly configured."""
        return bool(self._config.api_key)

    def _parse_json(self, content: str) -> dict[str, Any]:
        """Parse JSON response with error handling."""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMResponseError(
                f"Invalid JSON response: {e}",
                raw_response=content,
            ) from e


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def client(self):
        """Lazy-initialize synchronous client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._config.api_key)
        return self._client

    @property
    def async_client(self):
        """Lazy-initialize asynchronous client."""
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(api_key=self._config.api_key)
        return self._async_client

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Make a synchronous OpenAI completion request."""
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        if content is None:
            raise LLMResponseError("Empty response from OpenAI API")

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=content,
            model=self._config.model,
            provider=self.name,
            usage=usage,
            raw_response=response,
        )

    async def complete_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Make an asynchronous OpenAI completion request."""
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.async_client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        if content is None:
            raise LLMResponseError("Empty response from OpenAI API")

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=content,
            model=self._config.model,
            provider=self.name,
            usage=usage,
            raw_response=response,
        )


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def client(self):
        """Lazy-initialize synchronous client."""
        if self._client is None:
            try:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=self._config.api_key)
            except ImportError:
                raise LLMConfigurationError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    @property
    def async_client(self):
        """Lazy-initialize asynchronous client."""
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic

                self._async_client = AsyncAnthropic(api_key=self._config.api_key)
            except ImportError:
                raise LLMConfigurationError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._async_client

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Make a synchronous Anthropic completion request."""
        # Add JSON instruction to system prompt if needed
        effective_system = system_prompt
        if response_format == "json":
            effective_system += "\n\nRespond with valid JSON only."

        response = self.client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            system=effective_system,
            messages=[{"role": "user", "content": user_prompt}],
        )

        content = response.content[0].text if response.content else ""
        if not content:
            raise LLMResponseError("Empty response from Anthropic API")

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=content,
            model=self._config.model,
            provider=self.name,
            usage=usage,
            raw_response=response,
        )

    async def complete_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Make an asynchronous Anthropic completion request."""
        effective_system = system_prompt
        if response_format == "json":
            effective_system += "\n\nRespond with valid JSON only."

        response = await self.async_client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            system=effective_system,
            messages=[{"role": "user", "content": user_prompt}],
        )

        content = response.content[0].text if response.content else ""
        if not content:
            raise LLMResponseError("Empty response from Anthropic API")

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=content,
            model=self._config.model,
            provider=self.name,
            usage=usage,
            raw_response=response,
        )


# Provider registry
PROVIDERS: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def create_provider(settings: Settings) -> BaseProvider:
    """Factory function to create the appropriate LLM provider.

    Args:
        settings: Application settings

    Returns:
        Configured LLM provider instance

    Raises:
        LLMConfigurationError: If provider is not supported or misconfigured
    """
    provider_name = settings.llm_provider.lower()

    if provider_name not in PROVIDERS:
        raise LLMConfigurationError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: {', '.join(PROVIDERS.keys())}"
        )

    # Build provider config based on provider type
    if provider_name == "openai":
        config = ProviderConfig(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
        )
    elif provider_name == "anthropic":
        config = ProviderConfig(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=settings.anthropic_temperature,
            max_tokens=settings.anthropic_max_tokens,
        )
    else:
        raise LLMConfigurationError(f"No configuration for provider: {provider_name}")

    provider_class = PROVIDERS[provider_name]
    return provider_class(config)


def get_available_providers() -> list[str]:
    """Get list of available provider names."""
    return list(PROVIDERS.keys())
