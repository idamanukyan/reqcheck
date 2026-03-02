"""Observability module for metrics and tracing.

Provides a pluggable metrics interface that works with or without external
monitoring systems (Prometheus, StatsD, etc.). When no external system is
configured, metrics are collected in-memory for API exposure.

Usage:
    from reqcheck.core.observability import metrics

    # Record a counter
    metrics.increment("api.requests", tags={"endpoint": "/analyze"})

    # Record a timing
    with metrics.timer("llm.call_duration"):
        result = await call_llm()

    # Record a gauge
    metrics.gauge("circuit_breaker.state", 1, tags={"state": "open"})
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from reqcheck.core.logging import get_logger

logger = get_logger("observability")


@dataclass
class MetricValue:
    """A metric data point with timestamp."""

    name: str
    value: float
    timestamp: float
    metric_type: str  # counter, gauge, histogram
    tags: dict[str, str] = field(default_factory=dict)


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def increment(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        pass

    @abstractmethod
    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""
        pass

    @abstractmethod
    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        """Record a timing metric in milliseconds."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get all collected metrics (for in-memory backends)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all metrics."""
        pass


class InMemoryMetricsBackend(MetricsBackend):
    """In-memory metrics backend for when no external system is configured.

    Collects metrics that can be exposed via API endpoints.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_history: int = 1000):
        self._lock = threading.RLock()
        self._max_history = max_history

        # Counters: name -> total value
        self._counters: dict[str, float] = defaultdict(float)
        # Counter tags: name -> {tag_key: {tag_value: count}}
        self._counter_tags: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # Gauges: name -> current value
        self._gauges: dict[str, float] = {}
        self._gauge_tags: dict[str, dict[str, str]] = {}

        # Timings: name -> list of (timestamp, value_ms)
        self._timings: dict[str, list[tuple[float, float]]] = defaultdict(list)

        # Request tracking
        self._request_count = 0
        self._error_count = 0
        self._start_time = time.time()

    def increment(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value

            if tags:
                for key, tag_value in tags.items():
                    self._counter_tags[name][key][tag_value] += value

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""
        with self._lock:
            self._gauges[name] = value
            if tags:
                self._gauge_tags[name] = tags

    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        """Record a timing metric in milliseconds."""
        with self._lock:
            self._timings[name].append((time.time(), value_ms))

            # Prune old timings to prevent memory growth
            if len(self._timings[name]) > self._max_history:
                self._timings[name] = self._timings[name][-self._max_history:]

    def get_stats(self) -> dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            # Calculate timing statistics
            timing_stats = {}
            for name, values in self._timings.items():
                if values:
                    ms_values = [v[1] for v in values]
                    timing_stats[name] = {
                        "count": len(ms_values),
                        "min_ms": round(min(ms_values), 2),
                        "max_ms": round(max(ms_values), 2),
                        "avg_ms": round(sum(ms_values) / len(ms_values), 2),
                        "p50_ms": round(self._percentile(ms_values, 50), 2),
                        "p95_ms": round(self._percentile(ms_values, 95), 2),
                        "p99_ms": round(self._percentile(ms_values, 99), 2),
                    }

            return {
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "counters": dict(self._counters),
                "counter_breakdown": {
                    name: dict(tags)
                    for name, tags in self._counter_tags.items()
                },
                "gauges": dict(self._gauges),
                "timings": timing_stats,
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._counter_tags.clear()
            self._gauges.clear()
            self._gauge_tags.clear()
            self._timings.clear()
            self._request_count = 0
            self._error_count = 0
            self._start_time = time.time()

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        """Calculate the p-th percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]


class LoggingMetricsBackend(MetricsBackend):
    """Metrics backend that logs all metrics (for debugging)."""

    def __init__(self, log_level: str = "debug"):
        self._log_level = log_level
        self._log_fn = getattr(logger, log_level, logger.debug)

    def increment(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        self._log_fn(
            f"METRIC counter {name}",
            extra={"metric_name": name, "value": value, "tags": tags or {}},
        )

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        self._log_fn(
            f"METRIC gauge {name}",
            extra={"metric_name": name, "value": value, "tags": tags or {}},
        )

    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        self._log_fn(
            f"METRIC timing {name}",
            extra={"metric_name": name, "value_ms": round(value_ms, 2), "tags": tags or {}},
        )

    def get_stats(self) -> dict[str, Any]:
        return {"backend": "logging", "note": "Metrics are logged, not collected"}

    def reset(self) -> None:
        pass


class CompositeMetricsBackend(MetricsBackend):
    """Metrics backend that forwards to multiple backends."""

    def __init__(self, backends: list[MetricsBackend]):
        self._backends = backends

    def increment(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        for backend in self._backends:
            backend.increment(name, value, tags)

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        for backend in self._backends:
            backend.gauge(name, value, tags)

    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        for backend in self._backends:
            backend.timing(name, value_ms, tags)

    def get_stats(self) -> dict[str, Any]:
        # Return stats from the first backend that has them
        for backend in self._backends:
            stats = backend.get_stats()
            if stats.get("counters") or stats.get("timings"):
                return stats
        return {"backend": "composite", "backends": len(self._backends)}

    def reset(self) -> None:
        for backend in self._backends:
            backend.reset()


class Metrics:
    """Main metrics interface with timer context manager support."""

    def __init__(self, backend: MetricsBackend | None = None):
        self._backend = backend or InMemoryMetricsBackend()

    @property
    def backend(self) -> MetricsBackend:
        return self._backend

    def set_backend(self, backend: MetricsBackend) -> None:
        """Replace the metrics backend."""
        self._backend = backend

    def increment(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        self._backend.increment(name, value, tags)

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""
        self._backend.gauge(name, value, tags)

    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        """Record a timing metric in milliseconds."""
        self._backend.timing(name, value_ms, tags)

    @contextmanager
    def timer(
        self, name: str, tags: dict[str, str] | None = None
    ) -> Generator[None, None, None]:
        """Context manager for timing code blocks.

        Usage:
            with metrics.timer("operation.duration"):
                do_something()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.timing(name, elapsed_ms, tags)

    def get_stats(self) -> dict[str, Any]:
        """Get all collected metrics."""
        return self._backend.get_stats()

    def reset(self) -> None:
        """Reset all metrics."""
        self._backend.reset()


# Pre-defined metric names for consistency
class MetricNames:
    """Standard metric names used throughout the application."""

    # API metrics
    API_REQUESTS = "api.requests"
    API_ERRORS = "api.errors"
    API_LATENCY = "api.latency_ms"

    # Analysis metrics
    ANALYSIS_TOTAL = "analysis.total"
    ANALYSIS_ERRORS = "analysis.errors"
    ANALYSIS_DURATION = "analysis.duration_ms"
    ANALYSIS_ISSUES_FOUND = "analysis.issues_found"

    # LLM metrics
    LLM_CALLS = "llm.calls"
    LLM_ERRORS = "llm.errors"
    LLM_LATENCY = "llm.latency_ms"
    LLM_TOKENS_USED = "llm.tokens_used"
    LLM_CACHE_HITS = "llm.cache.hits"
    LLM_CACHE_MISSES = "llm.cache.misses"

    # Circuit breaker metrics
    CB_STATE_CHANGES = "circuit_breaker.state_changes"
    CB_REJECTIONS = "circuit_breaker.rejections"


# Global metrics instance
_metrics: Metrics | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> Metrics:
    """Get the global metrics instance."""
    global _metrics

    with _metrics_lock:
        if _metrics is None:
            _metrics = Metrics()
        return _metrics


def reset_metrics() -> None:
    """Reset the global metrics instance."""
    global _metrics

    with _metrics_lock:
        if _metrics is not None:
            _metrics.reset()


def configure_metrics(backend: MetricsBackend) -> None:
    """Configure the global metrics with a specific backend."""
    global _metrics

    with _metrics_lock:
        if _metrics is None:
            _metrics = Metrics(backend)
        else:
            _metrics.set_backend(backend)


# Convenience shortcut
metrics = get_metrics()
