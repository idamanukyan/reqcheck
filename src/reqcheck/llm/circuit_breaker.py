"""Circuit breaker pattern for LLM API calls.

Implements a circuit breaker to prevent cascading failures when the LLM
service is consistently failing. The breaker has three states:
- CLOSED: Normal operation, requests go through
- OPEN: Service is considered down, requests fail fast
- HALF_OPEN: Testing if service has recovered
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from reqcheck.core.logging import get_logger

logger = get_logger("llm.circuit_breaker")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, not calling service
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Number of consecutive failures before opening the circuit
    failure_threshold: int = 5

    # Time in seconds to wait before attempting recovery (half-open state)
    recovery_timeout: float = 30.0

    # Number of successful calls in half-open state to close the circuit
    success_threshold: int = 2

    # Time window in seconds to track failures (rolling window)
    failure_window: float = 60.0


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_rejected: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    last_state_change: float = field(default_factory=time.time)
    failure_timestamps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging/API responses."""
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_rejected": self.total_rejected,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "failures_in_window": len(self.failure_timestamps),
        }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and rejecting requests."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker is open. Service unavailable. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """Thread-safe circuit breaker for external service calls.

    Usage:
        breaker = CircuitBreaker(config)

        # Check before making call
        if breaker.can_execute():
            try:
                result = call_external_service()
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        else:
            raise CircuitBreakerOpen(breaker.time_until_retry())
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self._config = config or CircuitBreakerConfig()
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._stats.state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get a copy of current statistics."""
        with self._lock:
            return CircuitBreakerStats(
                state=self._stats.state,
                consecutive_failures=self._stats.consecutive_failures,
                consecutive_successes=self._stats.consecutive_successes,
                total_failures=self._stats.total_failures,
                total_successes=self._stats.total_successes,
                total_rejected=self._stats.total_rejected,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                last_state_change=self._stats.last_state_change,
                failure_timestamps=self._stats.failure_timestamps.copy(),
            )

    def can_execute(self) -> bool:
        """Check if a request can be executed.

        Returns True if the circuit allows the request to proceed.
        Automatically transitions from OPEN to HALF_OPEN after recovery timeout.
        """
        with self._lock:
            now = time.time()

            if self._stats.state == CircuitState.CLOSED:
                return True

            if self._stats.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                time_since_open = now - self._stats.last_state_change
                if time_since_open >= self._config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                    logger.info(
                        "Circuit breaker transitioning to half-open",
                        extra={
                            "time_since_open": round(time_since_open, 2),
                            "recovery_timeout": self._config.recovery_timeout,
                        },
                    )
                    return True
                return False

            # HALF_OPEN: allow requests through for testing
            return True

    def time_until_retry(self) -> float:
        """Get seconds until retry is allowed (when circuit is open)."""
        with self._lock:
            if self._stats.state != CircuitState.OPEN:
                return 0.0

            elapsed = time.time() - self._stats.last_state_change
            remaining = self._config.recovery_timeout - elapsed
            return max(0.0, remaining)

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            now = time.time()
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = now

            if self._stats.state == CircuitState.HALF_OPEN:
                # Check if we have enough successes to close the circuit
                if self._stats.consecutive_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        "Circuit breaker closed after successful recovery",
                        extra={
                            "consecutive_successes": self._stats.consecutive_successes,
                            "success_threshold": self._config.success_threshold,
                        },
                    )

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            now = time.time()
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = now

            # Update rolling window of failures
            self._prune_old_failures(now)
            self._stats.failure_timestamps.append(now)

            if self._stats.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker reopened after failure in half-open state",
                    extra={"consecutive_failures": self._stats.consecutive_failures},
                )

            elif self._stats.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                failures_in_window = len(self._stats.failure_timestamps)
                if (
                    self._stats.consecutive_failures >= self._config.failure_threshold
                    or failures_in_window >= self._config.failure_threshold
                ):
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "Circuit breaker opened due to failures",
                        extra={
                            "consecutive_failures": self._stats.consecutive_failures,
                            "failures_in_window": failures_in_window,
                            "failure_threshold": self._config.failure_threshold,
                            "recovery_timeout": self._config.recovery_timeout,
                        },
                    )

    def record_rejected(self) -> None:
        """Record a rejected request (when circuit is open)."""
        with self._lock:
            self._stats.total_rejected += 1

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        with self._lock:
            self._stats = CircuitBreakerStats()
            logger.info("Circuit breaker reset to initial state")

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state (internal, must hold lock)."""
        old_state = self._stats.state
        self._stats.state = new_state
        self._stats.last_state_change = time.time()

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0
            self._stats.failure_timestamps.clear()
        elif new_state == CircuitState.HALF_OPEN:
            self._stats.consecutive_successes = 0

        logger.debug(
            "Circuit breaker state transition",
            extra={
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        )

    def _prune_old_failures(self, now: float) -> None:
        """Remove failures outside the rolling window (internal, must hold lock)."""
        cutoff = now - self._config.failure_window
        self._stats.failure_timestamps = [
            ts for ts in self._stats.failure_timestamps if ts > cutoff
        ]


# Global circuit breaker instance for LLM calls
_llm_circuit_breaker: CircuitBreaker | None = None
_breaker_lock = threading.Lock()


def get_circuit_breaker(config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Get or create the global circuit breaker for LLM calls.

    Args:
        config: Optional configuration. Only used when creating new instance.

    Returns:
        The global CircuitBreaker instance.
    """
    global _llm_circuit_breaker

    with _breaker_lock:
        if _llm_circuit_breaker is None:
            _llm_circuit_breaker = CircuitBreaker(config)
            logger.info(
                "Circuit breaker initialized",
                extra={
                    "failure_threshold": (config or CircuitBreakerConfig()).failure_threshold,
                    "recovery_timeout": (config or CircuitBreakerConfig()).recovery_timeout,
                },
            )
        return _llm_circuit_breaker


def reset_circuit_breaker() -> None:
    """Reset the global circuit breaker (useful for testing)."""
    global _llm_circuit_breaker

    with _breaker_lock:
        if _llm_circuit_breaker is not None:
            _llm_circuit_breaker.reset()
        _llm_circuit_breaker = None
