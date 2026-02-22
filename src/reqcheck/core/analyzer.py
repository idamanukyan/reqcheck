"""Main analyzer orchestrator."""

from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from reqcheck.analyzers.ambiguity import AmbiguityAnalyzer
from reqcheck.analyzers.completeness import CompletenessAnalyzer
from reqcheck.analyzers.risk import RiskAnalyzer
from reqcheck.analyzers.testability import TestabilityAnalyzer
from reqcheck.core.config import Settings, get_settings
from reqcheck.core.exceptions import AnalysisTimeoutError, LLMClientError
from reqcheck.core.logging import get_logger, log_context, log_timing
from reqcheck.core.models import (
    AnalysisReport,
    Issue,
    IssueCategory,
    Requirement,
    ScoreBreakdown,
    Severity,
)
from reqcheck.llm.client import LLMClient

logger = get_logger("analyzer")

# Number of parallel workers for analyzer execution
ANALYZER_WORKERS = 4


class RequirementsAnalyzer:
    """
    Main orchestrator for requirements analysis.

    Coordinates multiple analyzers and produces a comprehensive report.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._llm_client: LLMClient | None = None

        # Initialize analyzers
        self._ambiguity_analyzer = AmbiguityAnalyzer(self._settings)
        self._completeness_analyzer = CompletenessAnalyzer(self._settings)
        self._testability_analyzer = TestabilityAnalyzer(self._settings)
        self._risk_analyzer = RiskAnalyzer(self._settings)

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-initialize shared LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient(self._settings)
        return self._llm_client

    def analyze(self, requirement: Requirement) -> AnalysisReport:
        """
        Perform comprehensive analysis of a requirement with timeout enforcement.

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
                "Starting requirement analysis",
                extra={
                    "title": requirement.title[:50],
                    "timeout_seconds": timeout,
                    "llm_enabled": self._settings.llm_available,
                },
            )

            # Run analysis with timeout enforcement
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._analyze_internal, requirement)
                try:
                    with log_timing(logger, "full_analysis"):
                        return future.result(timeout=timeout)
                except FuturesTimeoutError:
                    logger.error(
                        "Analysis timed out",
                        extra={"timeout_seconds": timeout},
                    )
                    raise AnalysisTimeoutError(timeout, requirement_id=requirement.id)

    def _analyze_internal(self, requirement: Requirement) -> AnalysisReport:
        """
        Internal analysis logic without timeout wrapper.

        This method contains the actual analysis pipeline.
        Analyzers are run in parallel for improved performance.
        """
        all_issues: list[Issue] = []
        scores = ScoreBreakdown()

        # Run all analyzers in parallel
        with log_timing(logger, "parallel_analysis"):
            results = self._run_analyzers_parallel(requirement)

        # Process ambiguity results
        if "ambiguity" in results:
            if results["ambiguity"]["success"]:
                issues, score = results["ambiguity"]["result"]
                all_issues.extend(issues)
                scores.ambiguity = score
                logger.debug(
                    "Ambiguity analysis complete",
                    extra={"issue_count": len(issues), "score": score},
                )
            else:
                logger.error(
                    "Ambiguity analysis failed",
                    extra={
                        "error": results["ambiguity"]["error"],
                        "error_type": results["ambiguity"]["error_type"],
                    },
                )

        # Process completeness results
        if "completeness" in results:
            if results["completeness"]["success"]:
                issues, score = results["completeness"]["result"]
                all_issues.extend(issues)
                scores.completeness = score
                logger.debug(
                    "Completeness analysis complete",
                    extra={"issue_count": len(issues), "score": score},
                )
            else:
                logger.error(
                    "Completeness analysis failed",
                    extra={
                        "error": results["completeness"]["error"],
                        "error_type": results["completeness"]["error_type"],
                    },
                )

        # Process testability results
        if "testability" in results:
            if results["testability"]["success"]:
                issues, score = results["testability"]["result"]
                all_issues.extend(issues)
                scores.testability = score
                logger.debug(
                    "Testability analysis complete",
                    extra={"issue_count": len(issues), "score": score},
                )
            else:
                logger.error(
                    "Testability analysis failed",
                    extra={
                        "error": results["testability"]["error"],
                        "error_type": results["testability"]["error_type"],
                    },
                )

        # Process risk results
        if "risk" in results:
            if results["risk"]["success"]:
                issues, _ = results["risk"]["result"]
                all_issues.extend(issues)
                logger.debug(
                    "Risk analysis complete",
                    extra={"signal_count": len(issues)},
                )
            else:
                logger.error(
                    "Risk analysis failed",
                    extra={
                        "error": results["risk"]["error"],
                        "error_type": results["risk"]["error_type"],
                    },
                )

        # Calculate overall score
        scores.calculate_overall()

        # Filter by minimum severity
        filtered_issues = self._filter_by_severity(all_issues)

        # Sort issues by severity
        sorted_issues = self._sort_issues(filtered_issues)

        # Generate summary and recommendations
        summary, recommendations = self._generate_summary(
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
            },
        )

        logger.info(
            "Analysis complete",
            extra={
                "blocker_count": report.blocker_count,
                "warning_count": report.warning_count,
                "suggestion_count": report.suggestion_count,
                "overall_score": report.scores.overall,
                "ready_for_dev": report.is_ready_for_dev,
            },
        )

        return report

    def _run_analyzers_parallel(
        self, requirement: Requirement
    ) -> dict[str, dict[str, Any]]:
        """
        Run all analyzers in parallel using ThreadPoolExecutor.

        Args:
            requirement: The requirement to analyze

        Returns:
            Dictionary mapping analyzer names to their results/errors
        """
        analyzers: dict[str, Callable[[Requirement], tuple[list[Issue], float]]] = {
            "ambiguity": self._ambiguity_analyzer.analyze,
            "completeness": self._completeness_analyzer.analyze,
            "testability": self._testability_analyzer.analyze,
            "risk": self._risk_analyzer.analyze,
        }

        results: dict[str, dict[str, Any]] = {}
        futures: dict[str, Future[tuple[list[Issue], float]]] = {}

        logger.debug(
            "Starting parallel analyzer execution",
            extra={"analyzer_count": len(analyzers), "workers": ANALYZER_WORKERS},
        )

        with ThreadPoolExecutor(max_workers=ANALYZER_WORKERS) as executor:
            # Submit all analyzer tasks
            for name, analyzer_func in analyzers.items():
                futures[name] = executor.submit(analyzer_func, requirement)

            # Collect results
            for name, future in futures.items():
                try:
                    result = future.result()
                    results[name] = {
                        "success": True,
                        "result": result,
                    }
                except Exception as e:
                    results[name] = {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

        successful = sum(1 for r in results.values() if r["success"])
        logger.debug(
            "Parallel analyzer execution complete",
            extra={"successful": successful, "total": len(analyzers)},
        )

        return results

    def _filter_by_severity(self, issues: list[Issue]) -> list[Issue]:
        """Filter issues by minimum severity setting."""
        severity_order = {
            Severity.BLOCKER: 0,
            Severity.WARNING: 1,
            Severity.SUGGESTION: 2,
        }
        min_level = severity_order.get(
            Severity(self._settings.min_severity), 2
        )

        return [
            issue
            for issue in issues
            if severity_order.get(issue.severity, 2) <= min_level
        ]

    def _sort_issues(self, issues: list[Issue]) -> list[Issue]:
        """Sort issues by severity (blockers first) then category."""
        severity_order = {
            Severity.BLOCKER: 0,
            Severity.WARNING: 1,
            Severity.SUGGESTION: 2,
        }
        category_order = {
            IssueCategory.COMPLETENESS: 0,
            IssueCategory.AMBIGUITY: 1,
            IssueCategory.TESTABILITY: 2,
            IssueCategory.RISK: 3,
        }

        return sorted(
            issues,
            key=lambda i: (
                severity_order.get(i.severity, 99),
                category_order.get(i.category, 99),
            ),
        )

    def _generate_summary(
        self,
        requirement: Requirement,
        issues: list[Issue],
        scores: ScoreBreakdown,
    ) -> tuple[str, list[str]]:
        """Generate executive summary and recommendations."""
        # Try LLM summary first
        if self._settings.llm_available:
            try:
                issues_text = "\n".join(
                    f"- [{i.severity.value}] {i.message}" for i in issues[:20]
                )
                response = self.llm_client.generate_summary(
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

        # Fallback to rule-based summary
        return self._generate_fallback_summary(issues, scores)

    def _generate_fallback_summary(
        self, issues: list[Issue], scores: ScoreBreakdown
    ) -> tuple[str, list[str]]:
        """Generate summary without LLM."""
        blocker_count = sum(1 for i in issues if i.severity == Severity.BLOCKER)
        warning_count = sum(1 for i in issues if i.severity == Severity.WARNING)

        if blocker_count > 0:
            status = "NOT ready for development"
            urgency = f"Found {blocker_count} blocker(s) that must be resolved."
        elif warning_count > 3:
            status = "needs improvement before development"
            urgency = f"Found {warning_count} warnings that should be addressed."
        elif warning_count > 0:
            status = "acceptable but could be improved"
            urgency = f"Found {warning_count} minor issues to consider."
        else:
            status = "ready for development"
            urgency = "No significant issues found."

        summary = (
            f"This requirement is {status}. {urgency} "
            f"Overall quality score: {scores.overall:.0%}."
        )

        # Generate recommendations
        recommendations = []
        if blocker_count > 0:
            recommendations.append("Resolve all blocker issues before starting development")

        if scores.completeness < 0.6:
            recommendations.append("Add missing acceptance criteria and error handling")

        if scores.ambiguity < 0.6:
            recommendations.append("Clarify vague terms and add specific requirements")

        if scores.testability < 0.6:
            recommendations.append(
                "Make acceptance criteria more testable with measurable outcomes"
            )

        # Category-specific recommendations
        by_category: dict[IssueCategory, int] = {}
        for issue in issues:
            by_category[issue.category] = by_category.get(issue.category, 0) + 1

        if by_category.get(IssueCategory.RISK, 0) > 2:
            recommendations.append("Schedule architecture/security review due to risk signals")

        if not recommendations:
            recommendations.append("Proceed with implementation")

        return summary, recommendations


def analyze_requirement(
    requirement: Requirement | dict[str, Any],
    settings: Settings | None = None,
) -> AnalysisReport:
    """
    Convenience function to analyze a single requirement.

    Args:
        requirement: Requirement object or dict with requirement data
        settings: Optional settings override

    Returns:
        Analysis report
    """
    if isinstance(requirement, dict):
        requirement = Requirement(**requirement)

    analyzer = RequirementsAnalyzer(settings)
    return analyzer.analyze(requirement)
