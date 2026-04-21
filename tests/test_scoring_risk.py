"""Tests for risk score inclusion in overall scoring."""

from reqcheck.core.constants import get_overall_score_weights
from reqcheck.core.models import ScoreBreakdown


class TestRiskInOverallScore:
    """Tests that risk score is included in the overall score calculation."""

    def test_overall_weights_include_risk(self):
        weights = get_overall_score_weights()
        assert "risk" in weights
        assert weights["risk"] > 0

    def test_weights_sum_to_one(self):
        weights = get_overall_score_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_score_breakdown_has_risk_field(self):
        breakdown = ScoreBreakdown()
        assert hasattr(breakdown, "risk")
        assert breakdown.risk == 0.0

    def test_calculate_overall_includes_risk(self):
        breakdown = ScoreBreakdown(
            ambiguity=1.0, completeness=1.0, testability=1.0, risk=0.0
        )
        score_with_zero_risk = breakdown.calculate_overall()

        breakdown2 = ScoreBreakdown(
            ambiguity=1.0, completeness=1.0, testability=1.0, risk=1.0
        )
        score_with_full_risk = breakdown2.calculate_overall()

        assert score_with_full_risk > score_with_zero_risk

    def test_calculate_overall_value(self):
        weights = get_overall_score_weights()
        breakdown = ScoreBreakdown(
            ambiguity=0.8, completeness=0.7, testability=0.9, risk=0.6
        )
        overall = breakdown.calculate_overall()

        expected = (
            0.8 * weights["ambiguity"]
            + 0.7 * weights["completeness"]
            + 0.9 * weights["testability"]
            + 0.6 * weights["risk"]
        )
        assert abs(overall - expected) < 1e-9
