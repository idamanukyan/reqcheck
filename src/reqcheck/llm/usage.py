"""Token usage and cost tracking for LLM API calls."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from reqcheck.core.logging import get_logger
from reqcheck.llm.providers import TokenUsage

logger = get_logger("llm.usage")


# Pricing per 1M tokens (as of 2024)
# Prices are in USD
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic models
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


@dataclass
class UsageStats:
    """Statistics for a single analysis session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    request_count: int = 0
    estimated_cost_usd: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens processed per second."""
        duration = self.duration_seconds
        if duration > 0:
            return self.total_tokens / duration
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "request_count": self.request_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "duration_seconds": round(self.duration_seconds, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


class UsageTracker:
    """Thread-safe tracker for LLM API usage and costs.

    Tracks token usage, estimates costs, and provides aggregated statistics.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current_session = UsageStats()
        self._total_stats = UsageStats()
        self._model_usage: dict[str, UsageStats] = {}

    def record_usage(
        self,
        usage: TokenUsage,
        model: str,
        provider: str,
    ) -> float:
        """Record token usage from an API call.

        Args:
            usage: Token usage from the API response
            model: Model name used
            provider: Provider name (openai, anthropic)

        Returns:
            Estimated cost in USD for this request
        """
        cost = self._calculate_cost(usage, model)

        with self._lock:
            # Update session stats
            self._current_session.prompt_tokens += usage.prompt_tokens
            self._current_session.completion_tokens += usage.completion_tokens
            self._current_session.total_tokens += usage.total_tokens
            self._current_session.cached_tokens += usage.cached_tokens
            self._current_session.request_count += 1
            self._current_session.estimated_cost_usd += cost

            # Update total stats
            self._total_stats.prompt_tokens += usage.prompt_tokens
            self._total_stats.completion_tokens += usage.completion_tokens
            self._total_stats.total_tokens += usage.total_tokens
            self._total_stats.cached_tokens += usage.cached_tokens
            self._total_stats.request_count += 1
            self._total_stats.estimated_cost_usd += cost

            # Update per-model stats
            if model not in self._model_usage:
                self._model_usage[model] = UsageStats()
            model_stats = self._model_usage[model]
            model_stats.prompt_tokens += usage.prompt_tokens
            model_stats.completion_tokens += usage.completion_tokens
            model_stats.total_tokens += usage.total_tokens
            model_stats.cached_tokens += usage.cached_tokens
            model_stats.request_count += 1
            model_stats.estimated_cost_usd += cost

        logger.debug(
            "Recorded LLM usage",
            extra={
                "model": model,
                "provider": provider,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "estimated_cost_usd": round(cost, 6),
            },
        )

        return cost

    def _calculate_cost(self, usage: TokenUsage, model: str) -> float:
        """Calculate estimated cost for token usage.

        Args:
            usage: Token usage
            model: Model name

        Returns:
            Estimated cost in USD
        """
        # Find pricing for model - prefer exact matches, then longer partial matches
        pricing = None
        best_match_len = 0

        for model_name, prices in MODEL_PRICING.items():
            # Exact match
            if model_name == model or model == model_name:
                pricing = prices
                break
            # Partial match - prefer longer matches to avoid "gpt-4o" matching "gpt-4o-mini"
            if model_name in model or model in model_name:
                match_len = len(model_name)
                if match_len > best_match_len:
                    best_match_len = match_len
                    pricing = prices

        if pricing is None:
            # Default to GPT-4o-mini pricing if unknown
            pricing = MODEL_PRICING.get("gpt-4o-mini", {"input": 0.15, "output": 0.60})

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_session_stats(self) -> UsageStats:
        """Get current session statistics."""
        with self._lock:
            # Return a copy
            stats = UsageStats(
                prompt_tokens=self._current_session.prompt_tokens,
                completion_tokens=self._current_session.completion_tokens,
                total_tokens=self._current_session.total_tokens,
                cached_tokens=self._current_session.cached_tokens,
                request_count=self._current_session.request_count,
                estimated_cost_usd=self._current_session.estimated_cost_usd,
                start_time=self._current_session.start_time,
            )
        return stats

    def get_total_stats(self) -> UsageStats:
        """Get total statistics across all sessions."""
        with self._lock:
            stats = UsageStats(
                prompt_tokens=self._total_stats.prompt_tokens,
                completion_tokens=self._total_stats.completion_tokens,
                total_tokens=self._total_stats.total_tokens,
                cached_tokens=self._total_stats.cached_tokens,
                request_count=self._total_stats.request_count,
                estimated_cost_usd=self._total_stats.estimated_cost_usd,
                start_time=self._total_stats.start_time,
            )
        return stats

    def get_model_stats(self) -> dict[str, dict[str, Any]]:
        """Get per-model usage statistics."""
        with self._lock:
            return {
                model: stats.to_dict() for model, stats in self._model_usage.items()
            }

    def reset_session(self) -> UsageStats:
        """Reset current session and return final stats.

        Returns:
            Final session statistics before reset
        """
        with self._lock:
            self._current_session.end_time = time.time()
            final_stats = self._current_session
            self._current_session = UsageStats()
        return final_stats

    def reset_all(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._current_session = UsageStats()
            self._total_stats = UsageStats()
            self._model_usage.clear()


# Global tracker instance
_global_tracker: UsageTracker | None = None


def get_usage_tracker() -> UsageTracker:
    """Get the global usage tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = UsageTracker()
    return _global_tracker


def reset_usage_tracker() -> None:
    """Reset the global usage tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset_all()
    _global_tracker = None


def record_usage(usage: TokenUsage, model: str, provider: str) -> float:
    """Convenience function to record usage to global tracker."""
    return get_usage_tracker().record_usage(usage, model, provider)


def get_session_stats() -> dict[str, Any]:
    """Get current session statistics as dict."""
    return get_usage_tracker().get_session_stats().to_dict()


def get_total_stats() -> dict[str, Any]:
    """Get total statistics as dict."""
    return get_usage_tracker().get_total_stats().to_dict()
