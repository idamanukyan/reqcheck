"""Common analysis utilities shared between sync and async analyzers.

This module contains logic that is identical between the synchronous and
asynchronous analyzer implementations to reduce code duplication.
"""

from typing import Any

from reqcheck.core.models import (
    Issue,
    IssueCategory,
    ScoreBreakdown,
    Severity,
)


# Severity ordering for filtering and sorting
SEVERITY_ORDER = {
    Severity.BLOCKER: 0,
    Severity.WARNING: 1,
    Severity.SUGGESTION: 2,
}

# Category ordering for sorting
CATEGORY_ORDER = {
    IssueCategory.COMPLETENESS: 0,
    IssueCategory.AMBIGUITY: 1,
    IssueCategory.TESTABILITY: 2,
    IssueCategory.RISK: 3,
}


def filter_issues_by_severity(
    issues: list[Issue],
    min_severity: str,
) -> list[Issue]:
    """Filter issues by minimum severity setting.

    Args:
        issues: List of issues to filter.
        min_severity: Minimum severity level to include ('blocker', 'warning', 'suggestion').

    Returns:
        Filtered list of issues meeting the minimum severity threshold.
    """
    min_level = SEVERITY_ORDER.get(Severity(min_severity), 2)

    return [
        issue
        for issue in issues
        if SEVERITY_ORDER.get(issue.severity, 2) <= min_level
    ]


def sort_issues(issues: list[Issue]) -> list[Issue]:
    """Sort issues by severity (blockers first) then category.

    Args:
        issues: List of issues to sort.

    Returns:
        Sorted list of issues.
    """
    return sorted(
        issues,
        key=lambda i: (
            SEVERITY_ORDER.get(i.severity, 99),
            CATEGORY_ORDER.get(i.category, 99),
        ),
    )


def generate_fallback_summary(
    issues: list[Issue],
    scores: ScoreBreakdown,
) -> tuple[str, list[str]]:
    """Generate summary without LLM.

    Args:
        issues: List of issues found during analysis.
        scores: Score breakdown from analysis.

    Returns:
        Tuple of (summary text, list of recommendations).
    """
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


def process_analyzer_result(
    name: str,
    result: dict[str, Any],
    all_issues: list[Issue],
    scores: ScoreBreakdown,
) -> None:
    """Process a single analyzer's result, updating issues and scores in place.

    Args:
        name: Name of the analyzer ('ambiguity', 'completeness', 'testability', 'risk').
        result: Result dictionary with 'success', 'result', 'error', 'error_type' keys.
        all_issues: List to append issues to (modified in place).
        scores: ScoreBreakdown to update (modified in place).
    """
    if not result.get("success"):
        return

    issues, score = result["result"]
    all_issues.extend(issues)

    # Update the appropriate score
    if name == "ambiguity":
        scores.ambiguity = score
    elif name == "completeness":
        scores.completeness = score
    elif name == "testability":
        scores.testability = score
    # Risk score is tracked separately but not included in overall score


def count_issues_by_severity(issues: list[Issue]) -> dict[str, int]:
    """Count issues by severity level.

    Args:
        issues: List of issues to count.

    Returns:
        Dictionary with 'blocker', 'warning', 'suggestion' counts.
    """
    return {
        "blocker": sum(1 for i in issues if i.severity == Severity.BLOCKER),
        "warning": sum(1 for i in issues if i.severity == Severity.WARNING),
        "suggestion": sum(1 for i in issues if i.severity == Severity.SUGGESTION),
    }
