"""Output formatters for analysis reports."""

import json
from typing import Any

from reqcheck.core.config import Settings, get_settings
from reqcheck.core.constants import (
    CHECKLIST_MESSAGE_MAX_LENGTH,
    CHECKLIST_TOP_ISSUES_COUNT,
    EVIDENCE_MAX_LENGTH,
    JSON_INDENT,
    QUALITY_PASS_THRESHOLD,
    SCORE_BAR_WIDTH,
)
from reqcheck.core.models import AnalysisReport, Issue, Severity


def format_json(report: AnalysisReport, indent: int = JSON_INDENT) -> str:
    """
    Format report as JSON.

    Args:
        report: Analysis report to format
        indent: JSON indentation level

    Returns:
        JSON string
    """
    return report.model_dump_json(indent=indent)


def format_markdown(
    report: AnalysisReport,
    settings: Settings | None = None,
) -> str:
    """
    Format report as Markdown.

    Args:
        report: Analysis report to format
        settings: Optional settings for customization

    Returns:
        Markdown string
    """
    settings = settings or get_settings()
    lines: list[str] = []

    # Header
    lines.append(f"# QA Analysis: {report.requirement_title}")
    lines.append("")

    # Status badge
    if report.is_ready_for_dev:
        lines.append("**Status:** Ready for Development")
    else:
        lines.append(f"**Status:** {report.blocker_count} Blocker(s) Found")
    lines.append("")

    # Summary
    if report.summary:
        lines.append("## Summary")
        lines.append(report.summary)
        lines.append("")

    # Scores
    lines.append("## Quality Scores")
    lines.append("")
    lines.append("| Dimension | Score |")
    lines.append("|-----------|-------|")
    lines.append(f"| Ambiguity | {_score_bar(report.scores.ambiguity)} |")
    lines.append(f"| Completeness | {_score_bar(report.scores.completeness)} |")
    lines.append(f"| Testability | {_score_bar(report.scores.testability)} |")
    lines.append(f"| **Overall** | **{report.scores.overall:.0%}** |")
    lines.append("")

    # Issues by severity
    if report.issues:
        lines.append("## Issues Found")
        lines.append("")

        # Group by severity
        blockers = [i for i in report.issues if i.severity == Severity.BLOCKER]
        warnings = [i for i in report.issues if i.severity == Severity.WARNING]
        suggestions = [i for i in report.issues if i.severity == Severity.SUGGESTION]

        if blockers:
            lines.append("### Blockers")
            lines.append("")
            for issue in blockers:
                lines.extend(_format_issue_md(issue, settings))
            lines.append("")

        if warnings:
            lines.append("### Warnings")
            lines.append("")
            for issue in warnings:
                lines.extend(_format_issue_md(issue, settings))
            lines.append("")

        if suggestions:
            lines.append("### Suggestions")
            lines.append("")
            for issue in suggestions:
                lines.extend(_format_issue_md(issue, settings))
            lines.append("")
    else:
        lines.append("## Issues Found")
        lines.append("")
        lines.append("No significant issues detected.")
        lines.append("")

    # Recommendations
    if report.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    return "\n".join(lines)


def _score_bar(score: float) -> str:
    """Generate a visual score bar."""
    filled = int(score * SCORE_BAR_WIDTH)
    empty = SCORE_BAR_WIDTH - filled
    bar = "█" * filled + "░" * empty
    return f"{bar} {score:.0%}"


def _format_issue_md(issue: Issue, settings: Settings) -> list[str]:
    """Format a single issue as Markdown."""
    lines = []

    # Issue header with location
    lines.append(f"- **[{issue.category.value.upper()}]** {issue.message}")

    # Location
    lines.append(f"  - Location: `{issue.location}`")

    # Evidence
    if settings.include_evidence and issue.evidence:
        evidence = issue.evidence[:EVIDENCE_MAX_LENGTH] + "..." if len(issue.evidence) > EVIDENCE_MAX_LENGTH else issue.evidence
        lines.append(f"  - Evidence: `{evidence}`")

    # Suggestion
    if settings.include_suggestions and issue.suggestion:
        lines.append(f"  - Fix: {issue.suggestion}")

    return lines


def format_summary(report: AnalysisReport) -> str:
    """
    Format a brief summary suitable for terminal output.

    Args:
        report: Analysis report to format

    Returns:
        Summary string
    """
    lines: list[str] = []

    # Title and status
    status_icon = "✓" if report.is_ready_for_dev else "✗"
    lines.append(f"{status_icon} {report.requirement_title}")
    lines.append("")

    # Issue counts
    lines.append(f"Issues: {report.blocker_count} blockers, {report.warning_count} warnings, {report.suggestion_count} suggestions")

    # Scores
    lines.append(
        f"Scores: Ambiguity {report.scores.ambiguity:.0%} | "
        f"Completeness {report.scores.completeness:.0%} | "
        f"Testability {report.scores.testability:.0%}"
    )
    lines.append(f"Overall: {report.scores.overall:.0%}")
    lines.append("")

    # Summary
    if report.summary:
        lines.append(report.summary)

    return "\n".join(lines)


def format_checklist(report: AnalysisReport) -> str:
    """
    Format report as a simple checklist.

    Args:
        report: Analysis report to format

    Returns:
        Checklist string
    """
    lines: list[str] = []

    lines.append(f"Requirement: {report.requirement_title}")
    lines.append("")

    # Quality checks
    checks = [
        ("Testability", report.scores.testability >= QUALITY_PASS_THRESHOLD, f"{report.scores.testability:.0%}"),
        ("Completeness", report.scores.completeness >= QUALITY_PASS_THRESHOLD, f"{report.scores.completeness:.0%}"),
        ("Clarity", report.scores.ambiguity >= QUALITY_PASS_THRESHOLD, f"{report.scores.ambiguity:.0%}"),
        ("No Blockers", report.blocker_count == 0, f"{report.blocker_count} found"),
    ]

    for name, passed, detail in checks:
        icon = "✓" if passed else "✗"
        lines.append(f"[{icon}] {name}: {detail}")

    lines.append("")

    # Top issues
    if report.issues:
        lines.append("Top Issues:")
        for issue in report.issues[:CHECKLIST_TOP_ISSUES_COUNT]:
            severity_icon = {"blocker": "!", "warning": "?", "suggestion": "i"}
            icon = severity_icon.get(issue.severity.value, "-")
            lines.append(f"  [{icon}] {issue.message[:CHECKLIST_MESSAGE_MAX_LENGTH]}")

    return "\n".join(lines)


def to_dict(report: AnalysisReport) -> dict[str, Any]:
    """
    Convert report to dictionary.

    Args:
        report: Analysis report

    Returns:
        Dictionary representation
    """
    return report.model_dump()
