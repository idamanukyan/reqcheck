"""Async main analyzer orchestrator with parallel execution."""

import asyncio
from typing import Any

from reqcheck.analyzers.async_analyzers import (
    AsyncAmbiguityAnalyzer,
    AsyncCompletenessAnalyzer,
    AsyncRiskAnalyzer,
    AsyncTestabilityAnalyzer,
)
from reqcheck.core.analysis_common import (
    filter_issues_by_severity,
    generate_fallback_summary,
    sort_issues,
)
from reqcheck.core.config import Settings, get_settings
from reqcheck.core.exceptions import AnalysisTimeoutError, LLMClientError
from reqcheck.core.logging import get_logger, log_context
from reqcheck.core.models import (
    AnalysisReport,
    Issue,
    Requirement,
    ScoreBreakdown,
)
from reqcheck.llm.async_client import AsyncLLMClient

logger = get_logger("async_analyzer")


class AsyncRequirementsAnalyzer:
    """
    Async orchestrator for requirements analysis.

    Runs all analyzers in parallel using asyncio for improved performance.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._llm_client: AsyncLLMClient | None = None

        # Initialize async analyzers
        self._ambiguity_analyzer = AsyncAmbiguityAnalyzer(self._settings)
        self._completeness_analyzer = AsyncCompletenessAnalyzer(self._settings)
        self._testability_analyzer = AsyncTestabilityAnalyzer(self._settings)
        self._risk_analyzer = AsyncRiskAnalyzer(self._settings)

    @property
    def llm_client(self) -> AsyncLLMClient:
        """Lazy-initialize shared async LLM client."""
        if self._llm_client is None:
            self._llm_client = AsyncLLMClient(self._settings)
        return self._llm_client

    async def analyze(self, requirement: Requirement) -> AnalysisReport:
        """
        Perform comprehensive async analysis of a requirement with timeout enforcement.

        Args:
            requirement: The requirement to analyze

        Returns:
            Complete analysis report with issues, scores, and recommendations

        Raises:
            AnalysisTimeoutError: If analysis exceeds the configured timeout
        """
        timeout = self._settings.analysis_timeout

        with log_context(requirement_id=requirement.id):
            logger.info(
                "Starting async requirement analysis",
                extra={
                    "title": requirement.title[:50],
                    "timeout_seconds": timeout,
                    "llm_enabled": self._settings.llm_available,
                },
            )

            try:
                return await asyncio.wait_for(
                    self._analyze_internal(requirement),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Analysis timed out",
                    extra={"timeout_seconds": timeout},
                )
                raise AnalysisTimeoutError(timeout, requirement_id=requirement.id)

    async def _analyze_internal(self, requirement: Requirement) -> AnalysisReport:
        """
        Internal analysis logic running all analyzers in parallel.

        Uses asyncio.gather for concurrent execution of all four analyzers.
        """
        all_issues: list[Issue] = []
        scores = ScoreBreakdown()

        # Run all analyzers in parallel using asyncio.gather
        logger.debug("Starting parallel async analyzer execution")

        results = await asyncio.gather(
            self._run_analyzer("ambiguity", self._ambiguity_analyzer.analyze, requirement),
            self._run_analyzer("completeness", self._completeness_analyzer.analyze, requirement),
            self._run_analyzer("testability", self._testability_analyzer.analyze, requirement),
            self._run_analyzer("risk", self._risk_analyzer.analyze, requirement),
            return_exceptions=True,
        )

        # Process results
        analyzer_names = ["ambiguity", "completeness", "testability", "risk"]
        for name, result in zip(analyzer_names, results):
            if isinstance(result, Exception):
                logger.error(
                    f"{name.capitalize()} analysis failed",
                    extra={
                        "error": str(result),
                        "error_type": type(result).__name__,
                    },
                )
                continue

            if result["success"]:
                issues, score = result["result"]
                all_issues.extend(issues)

                if name == "ambiguity":
                    scores.ambiguity = score
                elif name == "completeness":
                    scores.completeness = score
                elif name == "testability":
                    scores.testability = score
                # Risk score is not included in overall calculation

                logger.debug(
                    f"{name.capitalize()} analysis complete",
                    extra={"issue_count": len(issues), "score": score},
                )
            else:
                logger.error(
                    f"{name.capitalize()} analysis failed",
                    extra={
                        "error": result.get("error"),
                        "error_type": result.get("error_type"),
                    },
                )

        # Calculate overall score
        scores.calculate_overall()

        # Filter by minimum severity
        filtered_issues = filter_issues_by_severity(all_issues, self._settings.min_severity)

        # Sort issues by severity
        sorted_issues = sort_issues(filtered_issues)

        # Generate summary and recommendations
        summary, recommendations = await self._generate_summary(
            requirement, sorted_issues, scores
        )

        report = AnalysisReport(
            requirement_id=requirement.id,
            requirement_title=requirement.title,
            issues=sorted_issues,
            scores=scores,
            summary=summary,
            recommendations=recommendations,
            metadata={
                "total_issues": len(sorted_issues),
                "llm_enabled": self._settings.llm_available,
                "parallel_execution": True,
                "async_mode": True,
            },
        )

        logger.info(
            "Async analysis complete",
            extra={
                "blocker_count": report.blocker_count,
                "warning_count": report.warning_count,
                "suggestion_count": report.suggestion_count,
                "overall_score": report.scores.overall,
                "ready_for_dev": report.is_ready_for_dev,
            },
        )

        return report

    async def _run_analyzer(
        self,
        name: str,
        analyze_func,
        requirement: Requirement,
    ) -> dict[str, Any]:
        """Run a single analyzer with error handling."""
        try:
            result = await analyze_func(requirement)
            return {
                "success": True,
                "result": result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def _generate_summary(
        self,
        requirement: Requirement,
        issues: list[Issue],
        scores: ScoreBreakdown,
    ) -> tuple[str, list[str]]:
        """Generate executive summary and recommendations."""
        if self._settings.llm_available:
            try:
                issues_text = "\n".join(
                    f"- [{i.severity.value}] {i.message}" for i in issues[:20]
                )
                response = await self.llm_client.generate_summary(
                    title=requirement.title,
                    issues_summary=issues_text or "No issues found",
                    ambiguity_score=scores.ambiguity,
                    completeness_score=scores.completeness,
                    testability_score=scores.testability,
                )
                return (
                    response.get("summary", ""),
                    response.get("recommendations", []),
                )
            except LLMClientError as e:
                logger.warning(
                    "LLM summary generation failed, using fallback",
                    extra={"error": str(e), "error_type": type(e).__name__},
                )

        return generate_fallback_summary(issues, scores)


async def analyze_requirement_async(
    requirement: Requirement | dict[str, Any],
    settings: Settings | None = None,
) -> AnalysisReport:
    """
    Async convenience function to analyze a single requirement.

    Args:
        requirement: Requirement object or dict with requirement data
        settings: Optional settings override

    Returns:
        Analysis report
    """
    if isinstance(requirement, dict):
        requirement = Requirement(**requirement)

    analyzer = AsyncRequirementsAnalyzer(settings)
    return await analyzer.analyze(requirement)


async def analyze_requirements_batch(
    requirements: list[Requirement | dict[str, Any]],
    settings: Settings | None = None,
    max_concurrent: int = 5,
) -> list[AnalysisReport]:
    """
    Analyze multiple requirements in parallel.

    Args:
        requirements: List of requirements to analyze
        settings: Optional settings override
        max_concurrent: Maximum number of concurrent analyses

    Returns:
        List of analysis reports in the same order as input
    """
    analyzer = AsyncRequirementsAnalyzer(settings)

    # Convert dicts to Requirement objects
    reqs = [
        Requirement(**r) if isinstance(r, dict) else r
        for r in requirements
    ]

    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze_with_limit(req: Requirement) -> AnalysisReport:
        async with semaphore:
            return await analyzer.analyze(req)

    # Run all analyses in parallel with concurrency limit
    return await asyncio.gather(*[analyze_with_limit(req) for req in reqs])
