"""Constants and magic numbers used throughout reqcheck.

This module centralizes all numeric constants, thresholds, and weights
to improve maintainability and make calibration easier.
"""

# =============================================================================
# SCORING WEIGHTS
# =============================================================================

# Overall score calculation weights (must sum to 1.0)
SCORE_WEIGHT_AMBIGUITY = 0.30
SCORE_WEIGHT_COMPLETENESS = 0.35
SCORE_WEIGHT_TESTABILITY = 0.35

# Severity penalty weights for score estimation
SEVERITY_WEIGHT_BLOCKER = 0.15
SEVERITY_WEIGHT_WARNING = 0.08
SEVERITY_WEIGHT_SUGGESTION = 0.03
SEVERITY_WEIGHT_DEFAULT = 0.05

# Completeness-specific severity weights (higher impact)
COMPLETENESS_SEVERITY_WEIGHT_BLOCKER = 0.20
COMPLETENESS_SEVERITY_WEIGHT_WARNING = 0.10
COMPLETENESS_SEVERITY_WEIGHT_SUGGESTION = 0.05

# Risk-specific severity weights
RISK_SEVERITY_WEIGHT_BLOCKER = 0.20
RISK_SEVERITY_WEIGHT_WARNING = 0.10
RISK_SEVERITY_WEIGHT_SUGGESTION = 0.03


# =============================================================================
# SCORE DEFAULTS AND BASELINES
# =============================================================================

# Perfect/initial score
SCORE_PERFECT = 1.0

# Default score when LLM doesn't return a value
SCORE_DEFAULT_LLM_FALLBACK = 0.5

# Baseline score for rule-based estimation without LLM validation
SCORE_BASELINE_NO_LLM = 0.9

# Testability baseline score (lower due to uncertainty)
SCORE_BASELINE_TESTABILITY = 0.7

# Testability score when no acceptance criteria exist
SCORE_NO_ACCEPTANCE_CRITERIA = 0.2


# =============================================================================
# SCORING ADJUSTMENTS
# =============================================================================

# Text length threshold for reducing ambiguity penalty (longer text = more matches expected)
TEXT_LENGTH_LONG_THRESHOLD = 500

# Penalty reduction factor for longer requirements
PENALTY_REDUCTION_FACTOR_LONG_TEXT = 0.8

# Completeness penalties
PENALTY_MISSING_ACCEPTANCE_CRITERIA = 0.4
PENALTY_SHORT_DESCRIPTION = 0.2

# Testability bonus for having testable patterns
BONUS_TESTABLE_PATTERNS = 0.2

# Risk penalty for multiple risk factors
PENALTY_MULTIPLE_RISK_FACTORS = 0.1
RISK_FACTORS_HIGH_THRESHOLD = 3


# =============================================================================
# CONTENT LENGTH THRESHOLDS
# =============================================================================

# Minimum description length to be considered adequate
MIN_DESCRIPTION_LENGTH = 50

# Minimum acceptance criteria count
MIN_ACCEPTANCE_CRITERIA_COUNT = 1

# Minimum length for an acceptance criterion to be considered meaningful
MIN_ACCEPTANCE_CRITERION_LENGTH = 20

# Short criterion threshold for testability warnings
SHORT_CRITERION_THRESHOLD = 100

# Overlap ratio threshold for detecting title restatements
RESTATEMENT_OVERLAP_THRESHOLD = 0.7


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

# JSON indentation level
JSON_INDENT = 2

# Score bar visual width (number of characters)
SCORE_BAR_WIDTH = 10

# Maximum evidence length before truncation
EVIDENCE_MAX_LENGTH = 100

# Maximum issue message length in checklist format
CHECKLIST_MESSAGE_MAX_LENGTH = 80

# Number of top issues to show in checklist format
CHECKLIST_TOP_ISSUES_COUNT = 5

# Quality threshold for pass/fail in checklist (70%)
QUALITY_PASS_THRESHOLD = 0.7

# Maximum evidence matches to display in risk analysis
MAX_EVIDENCE_MATCHES = 5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_severity_weights() -> dict[str, float]:
    """Get standard severity weights as a dictionary."""
    return {
        "blocker": SEVERITY_WEIGHT_BLOCKER,
        "warning": SEVERITY_WEIGHT_WARNING,
        "suggestion": SEVERITY_WEIGHT_SUGGESTION,
    }


def get_completeness_severity_weights() -> dict[str, float]:
    """Get completeness-specific severity weights."""
    return {
        "blocker": COMPLETENESS_SEVERITY_WEIGHT_BLOCKER,
        "warning": COMPLETENESS_SEVERITY_WEIGHT_WARNING,
        "suggestion": COMPLETENESS_SEVERITY_WEIGHT_SUGGESTION,
    }


def get_risk_severity_weights() -> dict[str, float]:
    """Get risk-specific severity weights."""
    return {
        "blocker": RISK_SEVERITY_WEIGHT_BLOCKER,
        "warning": RISK_SEVERITY_WEIGHT_WARNING,
        "suggestion": RISK_SEVERITY_WEIGHT_SUGGESTION,
    }


def get_overall_score_weights() -> dict[str, float]:
    """Get overall score calculation weights."""
    return {
        "ambiguity": SCORE_WEIGHT_AMBIGUITY,
        "completeness": SCORE_WEIGHT_COMPLETENESS,
        "testability": SCORE_WEIGHT_TESTABILITY,
    }
