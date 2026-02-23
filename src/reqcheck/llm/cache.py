"""LLM response caching with TTL support."""

import hashlib
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class CacheEntry:
    """A cached LLM response with expiration."""

    value: dict[str, Any]
    expires_at: float

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.time() > self.expires_at


@dataclass
class LLMCache:
    """Thread-safe in-memory cache for LLM responses with TTL.

    This cache reduces API costs and latency by storing responses
    for identical requirement texts.
    """

    ttl_seconds: float = 3600.0  # Default 1 hour TTL
    max_size: int = 1000  # Maximum number of cached entries
    _cache: dict[str, CacheEntry] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)
    _hits: int = 0
    _misses: int = 0

    def _make_key(self, prompt_type: str, text: str) -> str:
        """Create a cache key from prompt type and text.

        Uses SHA-256 hash to handle long texts efficiently.
        """
        content = f"{prompt_type}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt_type: str, text: str) -> dict[str, Any] | None:
        """Retrieve a cached response if available and not expired.

        Args:
            prompt_type: Type of analysis (ambiguity, completeness, etc.)
            text: The requirement text that was analyzed

        Returns:
            Cached response dict or None if not found/expired
        """
        key = self._make_key(prompt_type, text)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.value

    def set(self, prompt_type: str, text: str, value: dict[str, Any]) -> None:
        """Store a response in the cache.

        Args:
            prompt_type: Type of analysis (ambiguity, completeness, etc.)
            text: The requirement text that was analyzed
            value: The LLM response to cache
        """
        key = self._make_key(prompt_type, text)
        expires_at = time.time() + self.ttl_seconds

        with self._lock:
            # Evict expired entries if we're at capacity
            if len(self._cache) >= self.max_size:
                self._evict_expired()

            # If still at capacity, evict oldest entries
            if len(self._cache) >= self.max_size:
                self._evict_oldest(len(self._cache) - self.max_size + 1)

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def _evict_expired(self) -> int:
        """Remove all expired entries. Must be called with lock held."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def _evict_oldest(self, count: int) -> None:
        """Remove the oldest entries. Must be called with lock held."""
        if count <= 0:
            return

        # Sort by expiration time (oldest first)
        sorted_keys = sorted(
            self._cache.keys(), key=lambda k: self._cache[k].expires_at
        )

        for key in sorted_keys[:count]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# Global cache instance (can be replaced for testing)
_global_cache: LLMCache | None = None


def get_cache(ttl_seconds: float = 3600.0, max_size: int = 1000) -> LLMCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache(ttl_seconds=ttl_seconds, max_size=max_size)
    return _global_cache


def reset_cache() -> None:
    """Reset the global cache (useful for testing)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = None
