"""Ambiguity detection analyzer."""

import logging

from reqcheck.analyzers.base import BaseAnalyzer
from reqcheck.core.models import Issue, IssueCategory, Requirement
from reqcheck.llm.client import LLMClientError

logger = logging.getLogger(__name__)


class AmbiguityAnalyzer(BaseAnalyzer):
    """Analyzer for detecting ambiguous language in requirements."""

    category = IssueCategory.AMBIGUITY

    def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """
        Analyze requirement for ambiguity issues.

        Checks for:
        - Vague terms (weasel words)
        - Ambiguous pronouns
        - Passive voice hiding actors
        - Unclear quantifiers
        - Temporal ambiguity
        """
        rule_issues: list[Issue] = []
        llm_issues: list[Issue] = []
        score = 1.0  # Start with perfect score

        # Run rule-based analysis
        if self._settings.enable_rule_based_analysis:
            rule_issues = self._run_rule_based_analysis(requirement)
            logger.debug(f"Rule-based analysis found {len(rule_issues)} ambiguity issues")

        # Run LLM analysis
        if self._settings.llm_available:
            try:
                response = self.llm_client.analyze_ambiguity(requirement.full_text)
                llm_issues = self._parse_llm_issues(response, self.category)
                score = response.get("ambiguity_score", 0.5)
                logger.debug(f"LLM analysis found {len(llm_issues)} ambiguity issues")
            except LLMClientError as e:
                logger.warning(f"LLM analysis failed: {e}")
                # Fall back to rule-based score estimation
                score = self._estimate_score_from_rules(rule_issues, requirement)

        # Merge and deduplicate
        all_issues = self._merge_issues(rule_issues, llm_issues)

        # If no LLM, estimate score from rules
        if not self._settings.llm_available:
            score = self._estimate_score_from_rules(all_issues, requirement)

        return all_issues, score

    def _estimate_score_from_rules(
        self, issues: list[Issue], requirement: Requirement
    ) -> float:
        """Estimate ambiguity score based on rule matches."""
        if not issues:
            return 0.9  # Assume good but not perfect without LLM validation

        # Weight by severity
        weights = {"blocker": 0.15, "warning": 0.08, "suggestion": 0.03}
        penalty = sum(weights.get(i.severity.value, 0.05) for i in issues)

        # Normalize by text length (more text = more potential matches)
        text_len = len(requirement.full_text)
        if text_len > 500:
            penalty *= 0.8  # Reduce penalty for longer requirements

        return max(0.0, min(1.0, 1.0 - penalty))
