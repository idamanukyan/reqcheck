"""Completeness checking analyzer."""

import logging

from reqcheck.analyzers.base import BaseAnalyzer
from reqcheck.core.constants import (
    MIN_ACCEPTANCE_CRITERIA_COUNT,
    MIN_ACCEPTANCE_CRITERION_LENGTH,
    MIN_DESCRIPTION_LENGTH,
    PENALTY_MISSING_ACCEPTANCE_CRITERIA,
    PENALTY_SHORT_DESCRIPTION,
    SCORE_DEFAULT_LLM_FALLBACK,
    SCORE_PERFECT,
    SEVERITY_WEIGHT_DEFAULT,
    get_completeness_severity_weights,
)
from reqcheck.core.models import Issue, IssueCategory, Requirement, Severity
from reqcheck.llm.client import LLMClientError

logger = logging.getLogger(__name__)


class CompletenessAnalyzer(BaseAnalyzer):
    """Analyzer for detecting completeness issues in requirements."""

    category = IssueCategory.COMPLETENESS

    # Structural checks that don't need LLM
    REQUIRED_SECTIONS = {
        "acceptance_criteria": "Acceptance criteria are missing",
        "description": "Description is empty or too short",
    }

    def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """
        Analyze requirement for completeness issues.

        Checks for:
        - Missing sections (description, acceptance criteria)
        - Missing error handling
        - Missing edge cases
        - Missing integration context
        - Open-ended language (escape hatches)
        """
        rule_issues: list[Issue] = []
        llm_issues: list[Issue] = []
        score = SCORE_PERFECT

        # Structural completeness checks
        structural_issues = self._check_structural_completeness(requirement)
        rule_issues.extend(structural_issues)

        # Run pattern-based analysis
        if self._settings.enable_rule_based_analysis:
            pattern_issues = self._run_rule_based_analysis(requirement)
            rule_issues.extend(pattern_issues)
            logger.debug(f"Rule-based analysis found {len(rule_issues)} completeness issues")

        # Run LLM analysis
        if self._settings.llm_available:
            try:
                response = self.llm_client.analyze_completeness(requirement.full_text)
                llm_issues = self._parse_llm_issues(response, self.category)
                score = response.get("completeness_score", SCORE_DEFAULT_LLM_FALLBACK)

                # Add missing sections as issues
                for section in response.get("missing_sections", []):
                    llm_issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            category=self.category,
                            location="missing",
                            message=f"Missing: {section}",
                            suggestion=f"Add section for {section}",
                        )
                    )

                logger.debug(f"LLM analysis found {len(llm_issues)} completeness issues")
            except LLMClientError as e:
                logger.warning(f"LLM analysis failed: {e}")
                score = self._estimate_score_from_rules(rule_issues, requirement)

        # Merge and deduplicate
        all_issues = self._merge_issues(rule_issues, llm_issues)

        if not self._settings.llm_available:
            score = self._estimate_score_from_rules(all_issues, requirement)

        return all_issues, score

    def _check_structural_completeness(self, requirement: Requirement) -> list[Issue]:
        """Check for basic structural completeness."""
        issues = []

        # Check description
        if len(requirement.description.strip()) < MIN_DESCRIPTION_LENGTH:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category=self.category,
                    location="description",
                    message="Description is too short or missing",
                    suggestion=(
                        f"Add a description of at least {MIN_DESCRIPTION_LENGTH} "
                        "characters explaining the requirement in detail"
                    ),
                    evidence=requirement.description[:50] if requirement.description else "(empty)",
                )
            )

        # Check acceptance criteria
        if len(requirement.acceptance_criteria) < MIN_ACCEPTANCE_CRITERIA_COUNT:
            issues.append(
                Issue(
                    severity=Severity.BLOCKER,
                    category=self.category,
                    location="acceptance_criteria",
                    message="No acceptance criteria defined",
                    suggestion=(
                        "Add at least one acceptance criterion that defines "
                        "when this requirement is satisfied"
                    ),
                )
            )

        # Check for extremely short acceptance criteria
        for i, ac in enumerate(requirement.acceptance_criteria):
            if len(ac.strip()) < MIN_ACCEPTANCE_CRITERION_LENGTH:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category=self.category,
                        location=f"acceptance_criteria[{i}]",
                        message="Acceptance criterion is too brief",
                        suggestion=(
                            "Expand to include specific conditions, "
                            "inputs, and expected outcomes"
                        ),
                        evidence=ac,
                    )
                )

        return issues

    def _estimate_score_from_rules(
        self, issues: list[Issue], requirement: Requirement
    ) -> float:
        """Estimate completeness score based on structural checks and rules."""
        base_score = SCORE_PERFECT

        # Major penalty for missing acceptance criteria
        if not requirement.acceptance_criteria:
            base_score -= PENALTY_MISSING_ACCEPTANCE_CRITERIA

        # Penalty for short description
        if len(requirement.description) < MIN_DESCRIPTION_LENGTH:
            base_score -= PENALTY_SHORT_DESCRIPTION

        # Additional penalties from issues
        weights = get_completeness_severity_weights()
        for issue in issues:
            base_score -= weights.get(issue.severity.value, SEVERITY_WEIGHT_DEFAULT)

        return max(0.0, min(1.0, base_score))
