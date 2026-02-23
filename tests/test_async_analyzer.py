"""Tests for async analyzer and caching functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reqcheck.core.async_analyzer import (
    AsyncRequirementsAnalyzer,
    analyze_requirement_async,
    analyze_requirements_batch,
)
from reqcheck.core.config import Settings
from reqcheck.core.exceptions import AnalysisTimeoutError
from reqcheck.core.models import Requirement
from reqcheck.llm.cache import LLMCache, get_cache, reset_cache


class TestLLMCache:
    """Tests for LLM response caching."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_cache()

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = LLMCache(ttl_seconds=3600)
        value = {"issues": [], "ambiguity_score": 0.8}

        cache.set("ambiguity", "test requirement text", value)
        result = cache.get("ambiguity", "test requirement text")

        assert result == value

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        cache = LLMCache(ttl_seconds=3600)

        result = cache.get("ambiguity", "nonexistent text")

        assert result is None

    def test_cache_different_prompt_types(self):
        """Test that different prompt types are cached separately."""
        cache = LLMCache(ttl_seconds=3600)
        text = "same requirement text"

        cache.set("ambiguity", text, {"type": "ambiguity"})
        cache.set("completeness", text, {"type": "completeness"})

        assert cache.get("ambiguity", text) == {"type": "ambiguity"}
        assert cache.get("completeness", text) == {"type": "completeness"}

    def test_cache_expiration(self):
        """Test that expired entries are not returned."""
        cache = LLMCache(ttl_seconds=0.01)  # Very short TTL
        value = {"issues": []}

        cache.set("ambiguity", "test", value)

        # Wait for expiration
        import time

        time.sleep(0.02)

        result = cache.get("ambiguity", "test")
        assert result is None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = LLMCache(ttl_seconds=3600)
        cache.set("ambiguity", "text1", {"score": 0.8})

        # Cache hit
        cache.get("ambiguity", "text1")
        # Cache miss
        cache.get("ambiguity", "text2")

        stats = cache.stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LLMCache(ttl_seconds=3600)
        cache.set("ambiguity", "text1", {"score": 0.8})
        cache.set("ambiguity", "text2", {"score": 0.9})

        cache.clear()

        assert cache.get("ambiguity", "text1") is None
        assert cache.get("ambiguity", "text2") is None
        assert cache.stats()["size"] == 0

    def test_cache_max_size_eviction(self):
        """Test that cache evicts entries when max size is reached."""
        cache = LLMCache(ttl_seconds=3600, max_size=3)

        # Fill cache beyond max size
        for i in range(5):
            cache.set("ambiguity", f"text{i}", {"index": i})

        # Should have at most max_size entries
        assert cache.stats()["size"] <= 3

    def test_global_cache_singleton(self):
        """Test that global cache returns same instance."""
        reset_cache()

        cache1 = get_cache(ttl_seconds=3600)
        cache2 = get_cache()

        assert cache1 is cache2


class TestAsyncRequirementsAnalyzer:
    """Tests for async requirements analyzer."""

    @pytest.fixture
    def settings(self):
        """Create settings with LLM disabled for testing."""
        return Settings(
            openai_api_key="",
            enable_llm_analysis=False,
            enable_rule_based_analysis=True,
        )

    @pytest.fixture
    def requirement(self):
        """Create a test requirement."""
        return Requirement(
            title="User Login",
            description="Users should be able to log in to the system properly",
            acceptance_criteria=[
                "User can enter email and password",
                "System validates credentials",
            ],
        )

    @pytest.mark.asyncio
    async def test_async_analyze_returns_report(self, settings, requirement):
        """Test that async analyzer returns a valid report."""
        analyzer = AsyncRequirementsAnalyzer(settings)

        report = await analyzer.analyze(requirement)

        assert report is not None
        assert report.requirement_id == requirement.id
        assert report.requirement_title == requirement.title
        assert 0.0 <= report.scores.overall <= 1.0

    @pytest.mark.asyncio
    async def test_async_analyze_detects_weasel_words(self, settings):
        """Test that async analyzer detects weasel words."""
        analyzer = AsyncRequirementsAnalyzer(settings)
        requirement = Requirement(
            title="Vague Feature",
            description="The system should handle things appropriately and efficiently.",
            acceptance_criteria=["Works correctly"],
        )

        report = await analyzer.analyze(requirement)

        # Should detect weasel words like "appropriately", "efficiently", "correctly"
        assert report.warning_count + report.suggestion_count > 0

    @pytest.mark.asyncio
    async def test_async_analyze_parallel_execution(self, settings, requirement):
        """Test that analyzers run in parallel."""
        analyzer = AsyncRequirementsAnalyzer(settings)

        # Run analysis and check metadata confirms parallel execution
        report = await analyzer.analyze(requirement)

        assert report.metadata.get("parallel_execution") is True
        assert report.metadata.get("async_mode") is True

    @pytest.mark.asyncio
    async def test_async_analyze_timeout(self):
        """Test that analysis times out properly."""
        settings = Settings(
            openai_api_key="",
            enable_llm_analysis=False,
            analysis_timeout=10,  # Minimum valid timeout
        )
        analyzer = AsyncRequirementsAnalyzer(settings)

        # Mock the internal analysis to be slow (longer than timeout)
        async def slow_analysis(*args, **kwargs):
            await asyncio.sleep(30)
            return ([], 0.8)

        requirement = Requirement(
            title="Test",
            description="Test description",
            acceptance_criteria=["Test criterion"],
        )

        with patch.object(
            analyzer._ambiguity_analyzer, "analyze", side_effect=slow_analysis
        ):
            with pytest.raises(AnalysisTimeoutError):
                await analyzer.analyze(requirement)


class TestAnalyzeRequirementAsync:
    """Tests for async convenience function."""

    @pytest.mark.asyncio
    async def test_analyze_requirement_with_dict(self):
        """Test analyzing requirement from dict."""
        settings = Settings(
            openai_api_key="",
            enable_llm_analysis=False,
        )

        requirement_dict = {
            "title": "Test Feature",
            "description": "Test description for the feature",
            "acceptance_criteria": ["First criterion", "Second criterion"],
        }

        report = await analyze_requirement_async(requirement_dict, settings=settings)

        assert report is not None
        assert report.requirement_title == "Test Feature"

    @pytest.mark.asyncio
    async def test_analyze_requirement_with_object(self):
        """Test analyzing requirement from Requirement object."""
        settings = Settings(
            openai_api_key="",
            enable_llm_analysis=False,
        )

        requirement = Requirement(
            title="Test Feature",
            description="Test description for the feature",
            acceptance_criteria=["First criterion", "Second criterion"],
        )

        report = await analyze_requirement_async(requirement, settings=settings)

        assert report is not None
        assert report.requirement_id == requirement.id


class TestAnalyzeRequirementsBatch:
    """Tests for batch analysis function."""

    @pytest.mark.asyncio
    async def test_batch_analyze_multiple_requirements(self):
        """Test analyzing multiple requirements in batch."""
        settings = Settings(
            openai_api_key="",
            enable_llm_analysis=False,
        )

        requirements = [
            Requirement(
                title=f"Feature {i}",
                description=f"Description for feature {i}",
                acceptance_criteria=[f"Criterion for feature {i}"],
            )
            for i in range(3)
        ]

        reports = await analyze_requirements_batch(requirements, settings=settings)

        assert len(reports) == 3
        for i, report in enumerate(reports):
            assert report.requirement_title == f"Feature {i}"

    @pytest.mark.asyncio
    async def test_batch_analyze_preserves_order(self):
        """Test that batch analysis preserves requirement order."""
        settings = Settings(
            openai_api_key="",
            enable_llm_analysis=False,
        )

        titles = ["First", "Second", "Third"]
        requirements = [
            Requirement(
                title=title,
                description=f"Description for {title}",
                acceptance_criteria=[f"Criterion for {title}"],
            )
            for title in titles
        ]

        reports = await analyze_requirements_batch(requirements, settings=settings)

        assert [r.requirement_title for r in reports] == titles

    @pytest.mark.asyncio
    async def test_batch_analyze_with_dicts(self):
        """Test batch analysis accepts dict input."""
        settings = Settings(
            openai_api_key="",
            enable_llm_analysis=False,
        )

        requirements = [
            {
                "title": "Dict Requirement",
                "description": "From a dictionary",
                "acceptance_criteria": ["Works with dicts"],
            }
        ]

        reports = await analyze_requirements_batch(requirements, settings=settings)

        assert len(reports) == 1
        assert reports[0].requirement_title == "Dict Requirement"

    @pytest.mark.asyncio
    async def test_batch_analyze_respects_concurrency_limit(self):
        """Test that batch analysis respects concurrency limits."""
        settings = Settings(
            openai_api_key="",
            enable_llm_analysis=False,
        )

        requirements = [
            Requirement(
                title=f"Feature {i}",
                description=f"Description {i}",
                acceptance_criteria=[f"Criterion {i}"],
            )
            for i in range(10)
        ]

        # Should complete without issues even with max_concurrent=2
        reports = await analyze_requirements_batch(
            requirements, settings=settings, max_concurrent=2
        )

        assert len(reports) == 10


class TestCacheIntegration:
    """Tests for cache integration with async analyzer."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_cache()

    @pytest.mark.asyncio
    async def test_cache_reduces_llm_calls(self):
        """Test that caching reduces redundant LLM calls."""
        settings = Settings(
            openai_api_key="test-key",
            enable_llm_analysis=True,
            llm_cache_enabled=True,
        )

        mock_response = {
            "issues": [],
            "ambiguity_score": 0.8,
        }

        # Same requirement analyzed twice should hit cache
        with patch(
            "reqcheck.llm.async_client.AsyncOpenAI"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = '{"issues": [], "ambiguity_score": 0.8}'

            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

            from reqcheck.llm.async_client import AsyncLLMClient

            client = AsyncLLMClient(settings)

            # First call - should hit API
            await client.analyze_ambiguity("test requirement")
            # Second call - should hit cache
            await client.analyze_ambiguity("test requirement")

            # API should only be called once
            assert mock_client.chat.completions.create.await_count == 1

            # Check cache stats
            stats = client.cache_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
