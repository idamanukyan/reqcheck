"""Unified score calculation utilities.

This module centralizes all score calculation logic to ensure consistency
across analyzers and reduce code duplication.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from reqcheck.core.constants import (
    BONUS_TESTABLE_PATTERNS,
    PENALTY_MISSING_ACCEPTANCE_CRITERIA,
    PENALTY_MULTIPLE_RISK_FACTORS,
    PENALTY_REDUCTION_FACTOR_LONG_TEXT,
    PENALTY_SHORT_DESCRIPTION,
    RISK_FACTORS_HIGH_THRESHOLD,
    SCORE_BASELINE_NO_LLM,
    SCORE_BASELINE_TESTABILITY,
    SCORE_NO_ACCEPTANCE_CRITERIA,
    SCORE_PERFECT,
    SEVERITY_WEIGHT_DEFAULT,
    TEXT_LENGTH_LONG_THRESHOLD,
    get_completeness_severity_weights,
    get_risk_severity_weights,
    get_severity_weights,
)
from reqcheck.core.models import Issue, IssueCategory, Requirement, Severity


class ScoringStrategy(Enum):
    """Available scoring strategies for different analyzer types."""

    STANDARD = "standard"  # Default severity-based scoring
    COMPLETENESS = "completeness"  # Completeness-specific weights
    TESTABILITY = "testability"  # Testability with bonus patterns
    RISK = "risk"  # Risk-specific scoring (inverted)


@dataclass
class ScoringContext:
    """Context information for score calculation."""

    issues: list[Issue]
    requirement: Requirement | None = None
    # Optional extras for specific strategies
    has_testable_patterns: bool = False
    testable_ratio: float = 0.0
    risk_factors: list[str] | None = None


class ScoreCalculator:
    """Unified score calculator for all analyzer types.

    Provides consistent score calculation with configurable strategies
    for different analyzer categories.

    Usage:
        calculator = ScoreCalculator(strategy=ScoringStrategy.STANDARD)
        score = calculator.calculate(issues, requirement)

        # Or with context for more control:
        context = ScoringContext(issues=issues, requirement=requirement)
        score = calculator.calculate_with_context(context)
    """

    def __init__(self, strategy: ScoringStrategy = ScoringStrategy.STANDARD):
        self._strategy = strategy
        self._weights = self._get_weights_for_strategy(strategy)

    @staticmethod
    def _get_weights_for_strategy(strategy: ScoringStrategy) -> dict[str, float]:
        """Get severity weights based on strategy."""
        if strategy == ScoringStrategy.COMPLETENESS:
            return get_completeness_severity_weights()
        elif strategy == ScoringStrategy.RISK:
            return get_risk_severity_weights()
        else:
            return get_severity_weights()

    def calculate(
        self,
        issues: list[Issue],
        requirement: Requirement | None = None,
    ) -> float:
        """Calculate score based on issues and optional requirement context.

        Args:
            issues: List of issues found during analysis.
            requirement: Optional requirement for context-aware scoring.

        Returns:
            Score from 0.0 (worst) to 1.0 (best).
        """
        context = ScoringContext(issues=issues, requirement=requirement)
        return self.calculate_with_context(context)

    def calculate_with_context(self, context: ScoringContext) -> float:
        """Calculate score with full context.

        Args:
            context: ScoringContext with issues, requirement, and extras.

        Returns:
            Score from 0.0 (worst) to 1.0 (best).
        """
        if self._strategy == ScoringStrategy.STANDARD:
            return self._calculate_standard(context)
        elif self._strategy == ScoringStrategy.COMPLETENESS:
            return self._calculate_completeness(context)
        elif self._strategy == ScoringStrategy.TESTABILITY:
            return self._calculate_testability(context)
        elif self._strategy == ScoringStrategy.RISK:
            return self._calculate_risk(context)
        else:
            return self._calculate_standard(context)

    def _calculate_penalty_from_issues(self, issues: list[Issue]) -> float:
        """Calculate total penalty from issues based on severity weights."""
        return sum(
            self._weights.get(issue.severity.value, SEVERITY_WEIGHT_DEFAULT)
            for issue in issues
        )

    def _calculate_standard(self, context: ScoringContext) -> float:
        """Standard severity-based scoring with text length normalization."""
        if not context.issues:
            return SCORE_BASELINE_NO_LLM

        penalty = self._calculate_penalty_from_issues(context.issues)

        # Normalize by text length if requirement is available
        if context.requirement:
            text_len = len(context.requirement.full_text)
            if text_len > TEXT_LENGTH_LONG_THRESHOLD:
                penalty *= PENALTY_REDUCTION_FACTOR_LONG_TEXT

        return self._clamp_score(SCORE_PERFECT - penalty)

    def _calculate_completeness(self, context: ScoringContext) -> float:
        """Completeness-specific scoring with structural penalties."""
        base_score = SCORE_PERFECT

        if context.requirement:
            # Major penalty for missing acceptance criteria
            if not context.requirement.acceptance_criteria:
                base_score -= PENALTY_MISSING_ACCEPTANCE_CRITERIA

            # Penalty for short description
            from reqcheck.core.constants import MIN_DESCRIPTION_LENGTH
            if len(context.requirement.description) < MIN_DESCRIPTION_LENGTH:
                base_score -= PENALTY_SHORT_DESCRIPTION

        # Additional penalties from issues
        base_score -= self._calculate_penalty_from_issues(context.issues)

        return self._clamp_score(base_score)

    def _calculate_testability(self, context: ScoringContext) -> float:
        """Testability scoring with pattern bonuses."""
        if context.requirement and not context.requirement.acceptance_criteria:
            return SCORE_NO_ACCEPTANCE_CRITERIA

        base_score = SCORE_BASELINE_TESTABILITY

        # Bonus for testable patterns
        if context.has_testable_patterns:
            base_score += context.testable_ratio * BONUS_TESTABLE_PATTERNS

        # Penalties from issues
        base_score -= self._calculate_penalty_from_issues(context.issues)

        return self._clamp_score(base_score)

    def _calculate_risk(self, context: ScoringContext) -> float:
        """Risk scoring (inverted: 1.0 = low risk, 0.0 = high risk)."""
        base_score = SCORE_PERFECT

        # Penalty based on issue severity
        base_score -= self._calculate_penalty_from_issues(context.issues)

        # Additional penalty for multiple risk factors
        if context.risk_factors and len(context.risk_factors) > RISK_FACTORS_HIGH_THRESHOLD:
            base_score -= PENALTY_MULTIPLE_RISK_FACTORS

        return self._clamp_score(base_score)

    @staticmethod
    def _clamp_score(score: float) -> float:
        """Clamp score to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, score))


# Pre-configured calculator instances for common use cases
_calculators: dict[ScoringStrategy, ScoreCalculator] = {}


def get_calculator(strategy: ScoringStrategy = ScoringStrategy.STANDARD) -> ScoreCalculator:
    """Get a cached calculator instance for the given strategy.

    Args:
        strategy: The scoring strategy to use.

    Returns:
        A ScoreCalculator configured for the strategy.
    """
    if strategy not in _calculators:
        _calculators[strategy] = ScoreCalculator(strategy)
    return _calculators[strategy]


# Convenience functions for common operations
def calculate_ambiguity_score(issues: list[Issue], requirement: Requirement) -> float:
    """Calculate ambiguity score using standard strategy."""
    return get_calculator(ScoringStrategy.STANDARD).calculate(issues, requirement)


def calculate_completeness_score(issues: list[Issue], requirement: Requirement) -> float:
    """Calculate completeness score using completeness strategy."""
    return get_calculator(ScoringStrategy.COMPLETENESS).calculate(issues, requirement)


def calculate_testability_score(
    issues: list[Issue],
    requirement: Requirement,
    testable_ratio: float = 0.0,
) -> float:
    """Calculate testability score using testability strategy.

    Args:
        issues: List of testability issues found.
        requirement: The requirement being analyzed.
        testable_ratio: Ratio of acceptance criteria with testable patterns (0.0-1.0).

    Returns:
        Testability score from 0.0 to 1.0.
    """
    calculator = get_calculator(ScoringStrategy.TESTABILITY)
    context = ScoringContext(
        issues=issues,
        requirement=requirement,
        has_testable_patterns=testable_ratio > 0,
        testable_ratio=testable_ratio,
    )
    return calculator.calculate_with_context(context)


def calculate_risk_score(
    issues: list[Issue],
    risk_factors: list[str] | None = None,
) -> float:
    """Calculate risk score using risk strategy.

    Args:
        issues: List of risk issues found.
        risk_factors: Optional list of identified risk factors.

    Returns:
        Risk score from 0.0 (high risk) to 1.0 (low risk).
    """
    calculator = get_calculator(ScoringStrategy.RISK)
    context = ScoringContext(
        issues=issues,
        risk_factors=risk_factors,
    )
    return calculator.calculate_with_context(context)
