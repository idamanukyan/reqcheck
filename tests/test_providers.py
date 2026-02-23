"""Tests for LLM provider abstraction and usage tracking."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from reqcheck.core.config import Settings
from reqcheck.core.exceptions import LLMConfigurationError
from reqcheck.llm.providers import (
    LLMResponse,
    TokenUsage,
    ProviderConfig,
    OpenAIProvider,
    AnthropicProvider,
    create_provider,
    get_available_providers,
)
from reqcheck.llm.usage import (
    UsageTracker,
    UsageStats,
    get_usage_tracker,
    reset_usage_tracker,
    MODEL_PRICING,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_basic_usage(self):
        """Test basic token usage tracking."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_total_tokens_provided(self):
        """Test when total_tokens is explicitly provided."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=200)

        # Should use provided total
        assert usage.total_tokens == 200

    def test_cached_tokens(self):
        """Test cached tokens tracking."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, cached_tokens=30)

        assert usage.cached_tokens == 30


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self):
        """Test creating an LLM response."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        response = LLMResponse(
            content='{"test": "value"}',
            model="gpt-4o-mini",
            provider="openai",
            usage=usage,
        )

        assert response.content == '{"test": "value"}'
        assert response.model == "gpt-4o-mini"
        assert response.provider == "openai"
        assert response.usage.total_tokens == 150


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProviderConfig(api_key="test-key", model="gpt-4o-mini")

        assert config.temperature == 0.3
        assert config.max_tokens == 2000
        assert config.extra_params == {}


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_provider_name(self):
        """Test provider name."""
        config = ProviderConfig(api_key="test-key", model="gpt-4o-mini")
        provider = OpenAIProvider(config)

        assert provider.name == "openai"

    def test_is_available_with_key(self):
        """Test is_available with API key."""
        config = ProviderConfig(api_key="test-key", model="gpt-4o-mini")
        provider = OpenAIProvider(config)

        assert provider.is_available() is True

    def test_is_available_without_key(self):
        """Test is_available without API key raises error."""
        with pytest.raises(LLMConfigurationError):
            ProviderConfig(api_key="", model="gpt-4o-mini")
            OpenAIProvider(ProviderConfig(api_key="", model="gpt-4o-mini"))


class TestCreateProvider:
    """Tests for provider factory function."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        settings = Settings(
            openai_api_key="test-key",
            llm_provider="openai",
        )
        provider = create_provider(settings)

        assert provider.name == "openai"
        assert isinstance(provider, OpenAIProvider)

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        settings = Settings(
            anthropic_api_key="test-key",
            llm_provider="anthropic",
        )
        provider = create_provider(settings)

        assert provider.name == "anthropic"
        assert isinstance(provider, AnthropicProvider)

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises error."""
        settings = Settings(
            openai_api_key="test-key",
        )
        # Manually set invalid provider
        settings.llm_provider = "invalid"  # type: ignore

        with pytest.raises(LLMConfigurationError):
            create_provider(settings)

    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = get_available_providers()

        assert "openai" in providers
        assert "anthropic" in providers


class TestUsageTracker:
    """Tests for usage tracking."""

    def setup_method(self):
        """Reset tracker before each test."""
        reset_usage_tracker()

    def test_record_usage(self):
        """Test recording usage."""
        tracker = UsageTracker()
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)

        cost = tracker.record_usage(usage, "gpt-4o-mini", "openai")

        assert cost > 0
        stats = tracker.get_session_stats()
        assert stats.prompt_tokens == 100
        assert stats.completion_tokens == 50
        assert stats.request_count == 1

    def test_accumulate_usage(self):
        """Test accumulating multiple requests."""
        tracker = UsageTracker()

        tracker.record_usage(TokenUsage(prompt_tokens=100, completion_tokens=50), "gpt-4o-mini", "openai")
        tracker.record_usage(TokenUsage(prompt_tokens=200, completion_tokens=100), "gpt-4o-mini", "openai")

        stats = tracker.get_session_stats()
        assert stats.prompt_tokens == 300
        assert stats.completion_tokens == 150
        assert stats.request_count == 2

    def test_per_model_stats(self):
        """Test per-model statistics."""
        tracker = UsageTracker()

        tracker.record_usage(TokenUsage(prompt_tokens=100, completion_tokens=50), "gpt-4o-mini", "openai")
        tracker.record_usage(TokenUsage(prompt_tokens=200, completion_tokens=100), "gpt-4o", "openai")

        model_stats = tracker.get_model_stats()
        assert "gpt-4o-mini" in model_stats
        assert "gpt-4o" in model_stats
        assert model_stats["gpt-4o-mini"]["prompt_tokens"] == 100
        assert model_stats["gpt-4o"]["prompt_tokens"] == 200

    def test_cost_calculation(self):
        """Test cost calculation for known model."""
        tracker = UsageTracker()

        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        cost = tracker.record_usage(usage, "gpt-4o-mini", "openai")

        # Expected: $0.15 + $0.60 = $0.75
        assert abs(cost - 0.75) < 0.01

    def test_reset_session(self):
        """Test resetting session stats."""
        tracker = UsageTracker()
        tracker.record_usage(TokenUsage(prompt_tokens=100, completion_tokens=50), "gpt-4o-mini", "openai")

        final = tracker.reset_session()
        assert final.prompt_tokens == 100

        # New session should be empty
        new_stats = tracker.get_session_stats()
        assert new_stats.prompt_tokens == 0

        # Total should still have the data
        total = tracker.get_total_stats()
        assert total.prompt_tokens == 100

    def test_global_tracker(self):
        """Test global tracker singleton."""
        tracker1 = get_usage_tracker()
        tracker2 = get_usage_tracker()

        assert tracker1 is tracker2

    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = UsageStats(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            request_count=1,
            estimated_cost_usd=0.001,
        )

        d = stats.to_dict()

        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["total_tokens"] == 150
        assert d["request_count"] == 1
        assert "duration_seconds" in d
        assert "tokens_per_second" in d


class TestModelPricing:
    """Tests for model pricing data."""

    def test_openai_models_have_pricing(self):
        """Test that common OpenAI models have pricing."""
        assert "gpt-4o" in MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING
        assert "gpt-3.5-turbo" in MODEL_PRICING

    def test_anthropic_models_have_pricing(self):
        """Test that common Anthropic models have pricing."""
        assert "claude-3-5-sonnet-20241022" in MODEL_PRICING
        assert "claude-3-5-haiku-20241022" in MODEL_PRICING

    def test_pricing_structure(self):
        """Test pricing structure has input and output."""
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing
            assert "output" in pricing
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0
