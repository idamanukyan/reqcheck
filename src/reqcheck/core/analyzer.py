"""Main analyzer orchestrator."""

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from reqcheck.analyzers.ambiguity import AmbiguityAnalyzer
from reqcheck.analyzers.completeness import CompletenessAnalyzer
from reqcheck.analyzers.risk import RiskAnalyzer
from reqcheck.analyzers.testability import TestabilityAnalyzer
from reqcheck.core.config import Settings, get_settings
from reqcheck.core.models import (
    AnalysisReport,
    Issue,
    IssueCategory,
    Requirement,
    ScoreBreakdown,
    Severity,
)
from reqcheck.llm.client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)


class AnalysisTimeoutError(Exception):
    """Raised when analysis exceeds the configured timeout."""

    def __init__(self, timeout_seconds: int, message: str | None = None):
        self.timeout_seconds = timeout_seconds
        self.message = message or f"Analysis timed out after {timeout_seconds} seconds"
        super().__init__(self.message)


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
        logger.info(
            f"Analyzing requirement: {requirement.id} - {requirement.title[:50]} "
            f"(timeout: {timeout}s)"
        )

        # Run analysis with timeout enforcement
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._analyze_internal, requirement)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.error(
                    f"Analysis timed out after {timeout}s for requirement: "
                    f"{requirement.id}"
                )
                raise AnalysisTimeoutError(timeout)

    def _analyze_internal(self, requirement: Requirement) -> AnalysisReport:
        """
        Internal analysis logic without timeout wrapper.

        This method contains the actual analysis pipeline.
        """
        all_issues: list[Issue] = []
        scores = ScoreBreakdown()

        # Run ambiguity analysis
        try:
            ambiguity_issues, scores.ambiguity = self._ambiguity_analyzer.analyze(
                requirement
            )
            all_issues.extend(ambiguity_issues)
            logger.debug(
                f"Ambiguity: {len(ambiguity_issues)} issues, "
                f"score: {scores.ambiguity:.2f}"
            )
        except Exception as e:
            logger.error(f"Ambiguity analysis failed: {e}")

        # Run completeness analysis
        try:
            completeness_issues, scores.completeness = (
                self._completeness_analyzer.analyze(requirement)
            )
            all_issues.extend(completeness_issues)
            logger.debug(
                f"Completeness: {len(completeness_issues)} issues, "
                f"score: {scores.completeness:.2f}"
            )
        except Exception as e:
            logger.error(f"Completeness analysis failed: {e}")

        # Run testability analysis
        try:
            testability_issues, scores.testability = (
                self._testability_analyzer.analyze(requirement)
            )
            all_issues.extend(testability_issues)
            logger.debug(
                f"Testability: {len(testability_issues)} issues, "
                f"score: {scores.testability:.2f}"
            )
        except Exception as e:
            logger.error(f"Testability analysis failed: {e}")

        # Run risk analysis
        try:
            risk_issues, _ = self._risk_analyzer.analyze(requirement)
            all_issues.extend(risk_issues)
            logger.debug(f"Risk: {len(risk_issues)} signals identified")
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")

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
            },
        )

        logger.info(
            f"Analysis complete: {report.blocker_count} blockers, "
            f"{report.warning_count} warnings, {report.suggestion_count} suggestions"
        )

        return report

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
                logger.warning(f"LLM summary generation failed: {e}")

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
