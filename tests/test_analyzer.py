"""Tests for the requirements analyzer."""

import pytest

from reqcheck.core.analyzer import RequirementsAnalyzer, analyze_requirement
from reqcheck.core.config import Settings
from reqcheck.core.models import Requirement, Severity


@pytest.fixture
def settings():
    """Create test settings with LLM disabled."""
    return Settings(
        enable_llm_analysis=False,
        enable_rule_based_analysis=True,
    )


@pytest.fixture
def analyzer(settings):
    """Create analyzer with test settings."""
    return RequirementsAnalyzer(settings)


class TestRequirementsAnalyzer:
    """Tests for RequirementsAnalyzer."""

    def test_analyze_poor_requirement(self, analyzer):
        """Test analysis of a poorly written requirement."""
        requirement = Requirement(
            title="User login",
            description="Users should be able to log in appropriately.",
            acceptance_criteria=["Login works correctly"],
        )

        report = analyzer.analyze(requirement)

        # Should find issues
        assert len(report.issues) > 0

        # Should have low scores due to ambiguity
        assert report.scores.ambiguity < 0.9
        assert report.scores.testability < 0.9

    def test_analyze_good_requirement(self, analyzer):
        """Test analysis of a well-written requirement."""
        requirement = Requirement(
            title="User Email Verification",
            description="After registration, users must verify their email within 24 hours.",
            acceptance_criteria=[
                "GIVEN a newly registered user WHEN they check email THEN they receive verification email within 60 seconds",
                "GIVEN an expired link WHEN clicked THEN user sees 'Link expired' error",
            ],
        )

        report = analyzer.analyze(requirement)

        # Should have higher scores
        assert report.scores.completeness >= 0.5
        assert report.scores.testability >= 0.5

    def test_detect_weasel_words(self, analyzer):
        """Test detection of weasel words."""
        requirement = Requirement(
            title="Feature",
            description="System should appropriately handle data in a timely manner.",
            acceptance_criteria=["Data is processed correctly"],
        )

        report = analyzer.analyze(requirement)

        # Should detect vague terms
        messages = [i.message.lower() for i in report.issues]
        assert any("vague" in m or "appropriately" in m for m in messages)

    def test_detect_missing_acceptance_criteria(self, analyzer):
        """Test detection of missing acceptance criteria."""
        requirement = Requirement(
            title="Some Feature",
            description="This feature does something important.",
            acceptance_criteria=[],  # No AC
        )

        report = analyzer.analyze(requirement)

        # Should flag missing AC as blocker
        blockers = [i for i in report.issues if i.severity == Severity.BLOCKER]
        assert len(blockers) > 0

    def test_detect_passive_voice(self, analyzer):
        """Test detection of passive voice hiding actors."""
        requirement = Requirement(
            title="Data Processing",
            description="The data will be validated and processed.",
            acceptance_criteria=["Results are displayed"],
        )

        report = analyzer.analyze(requirement)

        # Should detect passive voice
        messages = [i.message.lower() for i in report.issues]
        assert any("passive" in m for m in messages)

    def test_detect_risk_signals(self, analyzer):
        """Test detection of risk signals."""
        requirement = Requirement(
            title="Payment Integration",
            description="Integrate with payment gateway for credit card processing.",
            acceptance_criteria=["Payment is processed securely"],
        )

        report = analyzer.analyze(requirement)

        # Should identify security/financial risk
        risk_issues = [i for i in report.issues if i.category.value == "risk"]
        assert len(risk_issues) > 0

    def test_ready_for_dev_flag(self, analyzer):
        """Test is_ready_for_dev flag."""
        # Poor requirement with blockers
        poor_req = Requirement(
            title="Bad",
            description="",
            acceptance_criteria=[],
        )
        poor_report = analyzer.analyze(poor_req)
        assert not poor_report.is_ready_for_dev

    def test_analyze_requirement_function(self, settings):
        """Test convenience function."""
        report = analyze_requirement(
            {"title": "Test", "description": "Testing", "acceptance_criteria": ["Works"]},
            settings=settings,
        )
        assert report.requirement_title == "Test"


class TestPatternMatcher:
    """Tests for pattern matching."""

    def test_escape_hatches_detected(self, analyzer):
        """Test detection of escape hatch language."""
        requirement = Requirement(
            title="Support formats",
            description="Support various formats like PDF, etc.",
            acceptance_criteria=["Handles formats and more"],
        )

        report = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in report.issues)
        assert "etc" in messages or "open-ended" in messages

    def test_vague_quantifiers_detected(self, analyzer):
        """Test detection of vague quantifiers."""
        requirement = Requirement(
            title="Performance",
            description="Should handle many requests quickly.",
            acceptance_criteria=["Processes multiple items fast"],
        )

        report = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in report.issues)
        assert "vague" in messages or "quantifier" in messages


class TestScoring:
    """Tests for scoring logic."""

    def test_scores_in_valid_range(self, analyzer):
        """Test that all scores are between 0 and 1."""
        requirement = Requirement(
            title="Test",
            description="Test description",
            acceptance_criteria=["Test AC"],
        )

        report = analyzer.analyze(requirement)

        assert 0.0 <= report.scores.ambiguity <= 1.0
        assert 0.0 <= report.scores.completeness <= 1.0
        assert 0.0 <= report.scores.testability <= 1.0
        assert 0.0 <= report.scores.overall <= 1.0

    def test_overall_score_calculated(self, analyzer):
        """Test that overall score is calculated."""
        requirement = Requirement(
            title="Test",
            description="Test description",
            acceptance_criteria=["Test AC"],
        )

        report = analyzer.analyze(requirement)

        # Overall should be weighted average
        assert report.scores.overall > 0
