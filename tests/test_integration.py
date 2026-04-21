"""Integration tests for the full analysis pipeline.

These tests run the complete RequirementsAnalyzer pipeline without mocking
internal analyzers, verifying end-to-end behavior.
"""

import json
from pathlib import Path

import pytest

from reqcheck.core.analyzer import RequirementsAnalyzer, analyze_requirement
from reqcheck.core.config import Settings
from reqcheck.core.models import (
    AnalysisReport,
    IssueCategory,
    Requirement,
    Severity,
)
from reqcheck.output.formatters import (
    format_checklist,
    format_json,
    format_markdown,
    format_summary,
)


@pytest.fixture
def settings():
    """Settings with LLM disabled for deterministic integration tests."""
    return Settings(
        enable_llm_analysis=False,
        enable_rule_based_analysis=True,
    )


@pytest.fixture
def analyzer(settings):
    return RequirementsAnalyzer(settings)


class TestFullPipelinePoorRequirement:
    """End-to-end tests with a deliberately poor requirement."""

    POOR_REQ = Requirement(
        title="Login",
        description="Users should be able to log in properly and handle things appropriately.",
        acceptance_criteria=[],
    )

    def test_pipeline_returns_report(self, analyzer):
        report = analyzer.analyze(self.POOR_REQ)
        assert isinstance(report, AnalysisReport)

    def test_pipeline_finds_ambiguity_issues(self, analyzer):
        report = analyzer.analyze(self.POOR_REQ)
        ambiguity_issues = report.issues_by_category(IssueCategory.AMBIGUITY)
        assert len(ambiguity_issues) > 0

    def test_pipeline_finds_completeness_blockers(self, analyzer):
        report = analyzer.analyze(self.POOR_REQ)
        blockers = [i for i in report.issues if i.severity == Severity.BLOCKER]
        assert len(blockers) > 0
        assert not report.is_ready_for_dev

    def test_pipeline_scores_are_low(self, analyzer):
        report = analyzer.analyze(self.POOR_REQ)
        assert report.scores.completeness < 0.5
        assert report.scores.overall < 0.7

    def test_pipeline_generates_summary(self, analyzer):
        report = analyzer.analyze(self.POOR_REQ)
        assert report.summary
        assert len(report.recommendations) > 0


class TestFullPipelineGoodRequirement:
    """End-to-end tests with a well-written requirement."""

    GOOD_REQ = Requirement(
        title="User Email Verification",
        description=(
            "After registration, users must verify their email address before "
            "accessing the dashboard. An email containing a verification link is "
            "sent to the registered email address. The link expires after 24 hours."
        ),
        acceptance_criteria=[
            "GIVEN a newly registered user WHEN they check their email "
            "THEN they receive a verification email within 60 seconds",
            "GIVEN an expired verification link WHEN the user clicks it "
            "THEN they see an error message 'Link expired' and a button to request a new link",
            "GIVEN a user with unverified email WHEN they try to access the dashboard "
            "THEN they are redirected to a page showing 'Please verify your email'",
        ],
        type="story",
    )

    def test_pipeline_returns_report(self, analyzer):
        report = analyzer.analyze(self.GOOD_REQ)
        assert isinstance(report, AnalysisReport)

    def test_pipeline_scores_are_reasonable(self, analyzer):
        report = analyzer.analyze(self.GOOD_REQ)
        assert report.scores.completeness >= 0.5
        assert report.scores.testability >= 0.5
        assert report.scores.overall >= 0.4

    def test_pipeline_has_no_blockers(self, analyzer):
        report = analyzer.analyze(self.GOOD_REQ)
        assert report.is_ready_for_dev

    def test_pipeline_populates_all_scores(self, analyzer):
        report = analyzer.analyze(self.GOOD_REQ)
        assert report.scores.ambiguity > 0
        assert report.scores.completeness > 0
        assert report.scores.testability > 0


class TestFullPipelineRiskRequirement:
    """End-to-end tests for risk detection through the full pipeline."""

    RISK_REQ = Requirement(
        title="Payment Processing Integration",
        description=(
            "Integrate Stripe payment gateway for credit card processing. "
            "Store card details and handle PCI compliance. "
            "Third-party webhook handles refunds."
        ),
        acceptance_criteria=[
            "GIVEN a valid credit card WHEN user submits payment "
            "THEN charge is processed within 5 seconds",
            "GIVEN a failed payment WHEN webhook fires "
            "THEN admin is notified via email within 1 minute",
        ],
        type="story",
    )

    def test_pipeline_detects_risk_issues(self, analyzer):
        report = analyzer.analyze(self.RISK_REQ)
        risk_issues = report.issues_by_category(IssueCategory.RISK)
        assert len(risk_issues) > 0

    def test_pipeline_risk_score_populated(self, analyzer):
        report = analyzer.analyze(self.RISK_REQ)
        assert report.scores.risk < 1.0


class TestOutputFormatterIntegration:
    """Test that formatters work correctly with real analyzer output."""

    @pytest.fixture
    def report(self, analyzer):
        req = Requirement(
            title="Data Export",
            description="Users should be able to export data properly.",
            acceptance_criteria=["Export works"],
        )
        return analyzer.analyze(req)

    def test_json_output_is_valid_json(self, report):
        output = format_json(report)
        parsed = json.loads(output)
        assert "issues" in parsed
        assert "scores" in parsed

    def test_markdown_output_has_structure(self, report, settings):
        output = format_markdown(report, settings)
        assert "# QA Analysis:" in output
        assert "Quality Scores" in output
        assert "Overall" in output

    def test_summary_output_has_scores(self, report):
        output = format_summary(report)
        assert "Ambiguity" in output
        assert "Completeness" in output
        assert "Overall" in output

    def test_checklist_output_has_checks(self, report):
        output = format_checklist(report)
        assert "Testability" in output
        assert "Completeness" in output


class TestConvenienceFunction:
    """Test the analyze_requirement convenience function end-to-end."""

    def test_dict_input(self):
        report = analyze_requirement(
            {
                "title": "User Registration",
                "description": "New users can register with email and password.",
                "acceptance_criteria": [
                    "GIVEN a new visitor WHEN they fill the form THEN account is created"
                ],
            },
            settings=Settings(enable_llm_analysis=False),
        )
        assert isinstance(report, AnalysisReport)
        assert report.requirement_title == "User Registration"

    def test_requirement_input(self):
        req = Requirement(
            title="Search Feature",
            description="Users can search products by name.",
            acceptance_criteria=[
                "GIVEN a search term WHEN submitted THEN results shown in 2 seconds"
            ],
        )
        report = analyze_requirement(req, settings=Settings(enable_llm_analysis=False))
        assert isinstance(report, AnalysisReport)


class TestExampleFiles:
    """Test that the example JSON files work through the pipeline."""

    def test_good_requirement_example(self, analyzer):
        path = Path(__file__).parent.parent / "examples" / "good_requirement.json"
        if not path.exists():
            pytest.skip("Example file not found")

        with open(path) as f:
            data = json.load(f)
        req = Requirement(**data)
        report = analyzer.analyze(req)

        assert isinstance(report, AnalysisReport)
        assert report.is_ready_for_dev

    def test_sample_requirement_example(self, analyzer):
        path = Path(__file__).parent.parent / "examples" / "sample_requirement.json"
        if not path.exists():
            pytest.skip("Example file not found")

        with open(path) as f:
            data = json.load(f)
        req = Requirement(**data)
        report = analyzer.analyze(req)

        assert isinstance(report, AnalysisReport)
