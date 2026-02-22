"""Base analyzer interface."""

from abc import ABC, abstractmethod
from typing import Any

from reqcheck.core.config import Settings, get_settings
from reqcheck.core.models import Issue, IssueCategory, Requirement, Severity
from reqcheck.llm.client import LLMClient
from reqcheck.rules.patterns import PatternMatcher


class BaseAnalyzer(ABC):
    """Abstract base class for requirement analyzers."""

    category: IssueCategory

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        pattern_matcher: PatternMatcher | None = None,
    ):
        self._settings = settings or get_settings()
        self._llm_client = llm_client
        self._pattern_matcher = pattern_matcher or PatternMatcher()

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-initialize LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient(self._settings)
        return self._llm_client

    @abstractmethod
    def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """
        Analyze a requirement and return issues with a score.

        Returns:
            Tuple of (list of issues, score from 0.0 to 1.0)
        """
        pass

    def _run_rule_based_analysis(
        self, requirement: Requirement
    ) -> list[Issue]:
        """Run rule-based pattern matching for this category."""
        issues = []
        text = requirement.full_text

        for match in self._pattern_matcher.find_matches_by_category(text, self.category):
            # Determine location based on where the match was found
            location = self._determine_location(requirement, match.start_pos)
            issue = match.to_issue(location)
            issues.append(issue)

        return issues

    def _determine_location(self, requirement: Requirement, position: int) -> str:
        """Determine which field a position falls into."""
        title_len = len(requirement.title)
        desc_start = title_len + 1
        desc_end = desc_start + len(requirement.description)

        if position < title_len:
            return "title"
        elif position < desc_end:
            return "description"
        else:
            # Must be in acceptance criteria
            ac_text = "\n".join(requirement.acceptance_criteria)
            remaining_pos = position - desc_end - len("Acceptance Criteria:\n")
            current_pos = 0
            for i, ac in enumerate(requirement.acceptance_criteria):
                ac_len = len(f"- {ac}\n")
                if current_pos + ac_len > remaining_pos:
                    return f"acceptance_criteria[{i}]"
                current_pos += ac_len
            return "acceptance_criteria"

    def _parse_llm_issues(
        self, llm_response: dict[str, Any], default_category: IssueCategory
    ) -> list[Issue]:
        """Parse issues from LLM response."""
        issues = []
        for issue_data in llm_response.get("issues", []):
            try:
                severity_str = issue_data.get("severity", "warning").lower()
                severity = Severity(severity_str) if severity_str in [s.value for s in Severity] else Severity.WARNING

                issues.append(
                    Issue(
                        severity=severity,
                        category=default_category,
                        location=issue_data.get("location", "unknown"),
                        message=issue_data.get("message", ""),
                        suggestion=issue_data.get("suggestion", ""),
                        evidence=issue_data.get("evidence", ""),
                    )
                )
            except (KeyError, ValueError):
                # Skip malformed issues
                continue

        return issues

    def _deduplicate_issues(self, issues: list[Issue]) -> list[Issue]:
        """Remove duplicate issues based on message similarity."""
        seen_messages: set[str] = set()
        unique_issues = []

        for issue in issues:
            # Normalize message for comparison
            normalized = issue.message.lower().strip()
            if normalized not in seen_messages:
                seen_messages.add(normalized)
                unique_issues.append(issue)

        return unique_issues

    def _merge_issues(
        self, rule_issues: list[Issue], llm_issues: list[Issue]
    ) -> list[Issue]:
        """Merge rule-based and LLM issues, removing duplicates."""
        # LLM issues take precedence as they're more contextual
        all_issues = llm_issues + rule_issues
        return self._deduplicate_issues(all_issues)
