"""Testability analysis analyzer."""

import logging
import re

from reqcheck.analyzers.base import BaseAnalyzer
from reqcheck.core.models import Issue, IssueCategory, Requirement, Severity
from reqcheck.llm.client import LLMClientError

logger = logging.getLogger(__name__)


class TestabilityAnalyzer(BaseAnalyzer):
    """Analyzer for evaluating testability of requirements."""

    category = IssueCategory.TESTABILITY

    # Patterns indicating testable criteria
    TESTABLE_PATTERNS = [
        r"\bGIVEN\b.*\bWHEN\b.*\bTHEN\b",  # Gherkin format
        r"\bmust\s+(?:be|have|return|display|show)\b",
        r"\bshall\s+(?:be|have|return|display|show)\b",
        r"\bwill\s+(?:be|have|return|display|show)\b",
        r"\b\d+\s*(?:ms|seconds?|minutes?|hours?|days?)\b",  # Time constraints
        r"\b\d+\s*(?:MB|GB|KB|bytes?)\b",  # Size constraints
        r"\b(?:exactly|at least|at most|between)\s+\d+\b",  # Numeric constraints
        r"\berror\s+(?:code|message)\b",  # Error specifications
        r"\breturn(?:s|ed)?\s+(?:true|false|null|empty|\d+)\b",  # Return values
    ]

    # Patterns indicating untestable criteria
    UNTESTABLE_PATTERNS = [
        (r"\b(?:works?|functions?)\s+(?:correctly|properly|well)\b", "Vague success criterion"),
        (r"\buser[- ]friendly\b", "Subjective quality"),
        (r"\bintuitive(?:ly)?\b", "Subjective quality"),
        (r"\bseamless(?:ly)?\b", "Subjective quality"),
        (r"\beasy\s+to\s+(?:use|understand|read)\b", "Subjective quality"),
        (r"\bnice\s+(?:UI|interface|experience)\b", "Subjective quality"),
        (r"\bgood\s+(?:performance|UX|experience)\b", "Subjective quality"),
        (r"\bfast(?:er)?\b(?!\s*\d)", "Vague performance without metric"),
        (r"\bslow(?:er)?\b(?!\s*\d)", "Vague performance without metric"),
        (r"\bquick(?:ly)?\b(?!\s*\d)", "Vague timing without metric"),
        (r"\bresponsive\b(?!\s*\d)", "Vague performance without metric"),
        (r"\bsecure(?:ly)?\b(?!\s+(?:using|with|via))", "Vague security without specifics"),
    ]

    def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """
        Analyze requirement for testability issues.

        Checks for:
        - Unmeasurable outcomes
        - Subjective quality criteria
        - Missing pass/fail conditions
        - Vague acceptance criteria
        """
        rule_issues: list[Issue] = []
        llm_issues: list[Issue] = []
        score = 1.0

        # Check acceptance criteria testability
        ac_issues = self._check_acceptance_criteria_testability(requirement)
        rule_issues.extend(ac_issues)

        # Run pattern-based analysis
        if self._settings.enable_rule_based_analysis:
            pattern_issues = self._run_rule_based_analysis(requirement)
            rule_issues.extend(pattern_issues)
            logger.debug(f"Rule-based analysis found {len(rule_issues)} testability issues")

        # Run LLM analysis
        if self._settings.llm_available:
            try:
                response = self.llm_client.analyze_testability(requirement.full_text)
                llm_issues = self._parse_llm_issues(response, self.category)
                score = response.get("testability_score", 0.5)

                # Log suggested test scenarios for reference
                scenarios = response.get("suggested_test_scenarios", [])
                if scenarios:
                    logger.debug(f"Suggested test scenarios: {scenarios}")

                logger.debug(f"LLM analysis found {len(llm_issues)} testability issues")
            except LLMClientError as e:
                logger.warning(f"LLM analysis failed: {e}")
                score = self._estimate_score_from_rules(rule_issues, requirement)

        # Merge and deduplicate
        all_issues = self._merge_issues(rule_issues, llm_issues)

        if not self._settings.llm_available:
            score = self._estimate_score_from_rules(all_issues, requirement)

        return all_issues, score

    def _check_acceptance_criteria_testability(
        self, requirement: Requirement
    ) -> list[Issue]:
        """Check each acceptance criterion for testability."""
        issues = []

        for i, ac in enumerate(requirement.acceptance_criteria):
            location = f"acceptance_criteria[{i}]"

            # Check for testable patterns
            has_testable_pattern = any(
                re.search(pattern, ac, re.IGNORECASE)
                for pattern in self.TESTABLE_PATTERNS
            )

            # Check for untestable patterns
            for pattern, reason in self.UNTESTABLE_PATTERNS:
                match = re.search(pattern, ac, re.IGNORECASE)
                if match:
                    issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            category=self.category,
                            location=location,
                            message=f"Untestable criterion: {reason}",
                            suggestion="Replace with measurable, objective criteria",
                            evidence=match.group(),
                        )
                    )

            # If no testable patterns found and it's a short criterion
            if not has_testable_pattern and len(ac) < 100:
                # Check if it's just a restatement
                if self._is_restatement(ac, requirement.title):
                    issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            category=self.category,
                            location=location,
                            message="Acceptance criterion appears to restate the title without adding testable detail",
                            suggestion="Add specific conditions, inputs, and expected outputs",
                            evidence=ac,
                        )
                    )

        return issues

    def _is_restatement(self, ac: str, title: str) -> bool:
        """Check if acceptance criterion is just restating the title."""
        # Normalize both strings
        ac_words = set(re.findall(r"\w+", ac.lower()))
        title_words = set(re.findall(r"\w+", title.lower()))

        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "will", "should", "must", "can", "be", "to", "and", "or", "it"}
        ac_words -= stop_words
        title_words -= stop_words

        if not ac_words or not title_words:
            return False

        # Check overlap
        overlap = len(ac_words & title_words) / len(ac_words)
        return overlap > 0.7

    def _estimate_score_from_rules(
        self, issues: list[Issue], requirement: Requirement
    ) -> float:
        """Estimate testability score based on patterns and issues."""
        if not requirement.acceptance_criteria:
            return 0.2  # Very low score without AC

        base_score = 0.7  # Start with reasonable baseline

        # Bonus for testable patterns in AC
        testable_count = 0
        for ac in requirement.acceptance_criteria:
            if any(
                re.search(pattern, ac, re.IGNORECASE)
                for pattern in self.TESTABLE_PATTERNS
            ):
                testable_count += 1

        if requirement.acceptance_criteria:
            testable_ratio = testable_count / len(requirement.acceptance_criteria)
            base_score += testable_ratio * 0.2

        # Penalties from issues
        weights = {"blocker": 0.15, "warning": 0.08, "suggestion": 0.03}
        for issue in issues:
            base_score -= weights.get(issue.severity.value, 0.05)

        return max(0.0, min(1.0, base_score))
