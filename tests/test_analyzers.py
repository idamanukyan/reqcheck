"""Tests for individual analyzer modules."""

import pytest

from reqcheck.analyzers.ambiguity import AmbiguityAnalyzer
from reqcheck.analyzers.completeness import CompletenessAnalyzer
from reqcheck.analyzers.risk import RiskAnalyzer
from reqcheck.analyzers.testability import TestabilityAnalyzer
from reqcheck.core.config import Settings
from reqcheck.core.models import IssueCategory, Requirement, Severity


@pytest.fixture
def settings():
    """Create test settings with LLM disabled."""
    return Settings(
        enable_llm_analysis=False,
        enable_rule_based_analysis=True,
    )


# =============================================================================
# Ambiguity Analyzer Tests
# =============================================================================


class TestAmbiguityAnalyzer:
    """Tests for AmbiguityAnalyzer."""

    @pytest.fixture
    def analyzer(self, settings):
        """Create ambiguity analyzer with test settings."""
        return AmbiguityAnalyzer(settings)

    def test_detects_weasel_words(self, analyzer):
        """Test detection of weasel words like 'appropriately', 'properly'."""
        requirement = Requirement(
            title="Data Processing",
            description="The system should appropriately handle data.",
            acceptance_criteria=["Data is processed properly"],
        )

        issues, score = analyzer.analyze(requirement)

        messages = [i.message.lower() for i in issues]
        assert any("appropriately" in m or "vague" in m for m in messages)
        assert score < 1.0

    def test_detects_passive_voice(self, analyzer):
        """Test detection of passive voice hiding actors."""
        requirement = Requirement(
            title="Report Generation",
            description="The report will be generated and sent.",
            acceptance_criteria=["Data is validated and stored"],
        )

        issues, score = analyzer.analyze(requirement)

        messages = [i.message.lower() for i in issues]
        assert any("passive" in m for m in messages)

    def test_detects_vague_quantifiers(self, analyzer):
        """Test detection of vague quantifiers like 'many', 'some', 'few'."""
        requirement = Requirement(
            title="Performance",
            description="The system should handle many concurrent users.",
            acceptance_criteria=["Some requests complete quickly"],
        )

        issues, score = analyzer.analyze(requirement)

        messages = [i.message.lower() for i in issues]
        assert any("vague" in m or "quantifier" in m or "many" in m for m in messages)

    def test_detects_temporal_ambiguity(self, analyzer):
        """Test detection of temporal ambiguity like 'immediately', 'soon'."""
        requirement = Requirement(
            title="Notification System",
            description="Send notifications immediately to users.",
            acceptance_criteria=["Notifications arrive soon after trigger"],
        )

        issues, score = analyzer.analyze(requirement)

        # Should detect temporal ambiguity
        messages = " ".join(i.message.lower() for i in issues)
        assert len(issues) > 0
        assert "timing" in messages or "vague" in messages or "immediately" in messages or "soon" in messages

    def test_clear_requirement_has_high_score(self, analyzer):
        """Test that clear requirements get high scores."""
        requirement = Requirement(
            title="User Authentication",
            description="Users authenticate using email and password.",
            acceptance_criteria=[
                "GIVEN valid credentials WHEN user logs in THEN session is created",
                "GIVEN invalid password WHEN user logs in THEN error 401 is returned",
            ],
        )

        issues, score = analyzer.analyze(requirement)

        # Should have few or no ambiguity issues
        ambiguity_issues = [i for i in issues if i.category == IssueCategory.AMBIGUITY]
        assert len(ambiguity_issues) <= 2
        assert score >= 0.7

    def test_empty_description_handled(self, analyzer):
        """Test that empty description doesn't crash analyzer."""
        requirement = Requirement(
            title="Feature",
            description="",
            acceptance_criteria=["It works"],
        )

        issues, score = analyzer.analyze(requirement)

        # Should complete without error
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_all_issues_have_correct_category(self, analyzer):
        """Test that all returned issues have AMBIGUITY category."""
        requirement = Requirement(
            title="Vague Feature",
            description="The system should appropriately handle things properly.",
            acceptance_criteria=["Works correctly"],
        )

        issues, _ = analyzer.analyze(requirement)

        for issue in issues:
            assert issue.category == IssueCategory.AMBIGUITY


# =============================================================================
# Completeness Analyzer Tests
# =============================================================================


class TestCompletenessAnalyzer:
    """Tests for CompletenessAnalyzer."""

    @pytest.fixture
    def analyzer(self, settings):
        """Create completeness analyzer with test settings."""
        return CompletenessAnalyzer(settings)

    def test_detects_missing_acceptance_criteria(self, analyzer):
        """Test detection of missing acceptance criteria."""
        requirement = Requirement(
            title="User Login",
            description="Users should be able to log in to the system.",
            acceptance_criteria=[],  # No AC
        )

        issues, score = analyzer.analyze(requirement)

        # Should flag missing AC as blocker
        blockers = [i for i in issues if i.severity == Severity.BLOCKER]
        assert len(blockers) > 0
        assert any("acceptance criteria" in i.message.lower() for i in blockers)
        assert score < 0.5

    def test_detects_short_description(self, analyzer):
        """Test detection of too-short description."""
        requirement = Requirement(
            title="Feature",
            description="Do it.",  # Too short
            acceptance_criteria=["It works as expected"],
        )

        issues, score = analyzer.analyze(requirement)

        messages = [i.message.lower() for i in issues]
        assert any("description" in m and ("short" in m or "missing" in m) for m in messages)

    def test_detects_brief_acceptance_criteria(self, analyzer):
        """Test detection of too-brief acceptance criteria."""
        requirement = Requirement(
            title="Data Export",
            description="Allow users to export their data in various formats.",
            acceptance_criteria=["Works", "Done"],  # Too brief
        )

        issues, score = analyzer.analyze(requirement)

        messages = [i.message.lower() for i in issues]
        assert any("brief" in m or "short" in m for m in messages)

    def test_complete_requirement_has_high_score(self, analyzer):
        """Test that complete requirements get high scores."""
        requirement = Requirement(
            title="User Registration",
            description=(
                "New users can create an account by providing email, password, "
                "and optional profile information. The system validates input, "
                "checks for duplicate emails, and sends a verification email."
            ),
            acceptance_criteria=[
                "GIVEN a valid email and password WHEN registering THEN account is created",
                "GIVEN an existing email WHEN registering THEN error is shown",
                "GIVEN invalid password format WHEN registering THEN validation error appears",
                "After registration, user receives verification email within 60 seconds",
            ],
        )

        issues, score = analyzer.analyze(requirement)

        # Should have high completeness score
        assert score >= 0.6
        # Should not have blocker issues
        blockers = [i for i in issues if i.severity == Severity.BLOCKER]
        assert len(blockers) == 0

    def test_detects_open_ended_language(self, analyzer):
        """Test detection of open-ended language that suggests incompleteness."""
        requirement = Requirement(
            title="Error Handling",
            description="Handle errors appropriately, etc.",
            acceptance_criteria=["Errors are handled and more"],
        )

        issues, _ = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert "etc" in messages or "open" in messages or "escape" in messages

    def test_all_issues_have_correct_category(self, analyzer):
        """Test that all returned issues have COMPLETENESS category."""
        requirement = Requirement(
            title="Incomplete",
            description="Short",
            acceptance_criteria=[],
        )

        issues, _ = analyzer.analyze(requirement)

        for issue in issues:
            assert issue.category == IssueCategory.COMPLETENESS


# =============================================================================
# Testability Analyzer Tests
# =============================================================================


class TestTestabilityAnalyzer:
    """Tests for TestabilityAnalyzer."""

    @pytest.fixture
    def analyzer(self, settings):
        """Create testability analyzer with test settings."""
        return TestabilityAnalyzer(settings)

    def test_detects_subjective_criteria(self, analyzer):
        """Test detection of subjective/untestable criteria."""
        requirement = Requirement(
            title="User Interface",
            description="Create a user-friendly interface.",
            acceptance_criteria=[
                "The UI is intuitive",
                "The experience is seamless",
                "The interface is nice",
            ],
        )

        issues, score = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert "subjective" in messages or "untestable" in messages
        assert score < 0.8

    def test_detects_vague_performance_criteria(self, analyzer):
        """Test detection of vague performance criteria without metrics."""
        requirement = Requirement(
            title="Performance",
            description="The system should be fast and responsive.",
            acceptance_criteria=[
                "Pages load quickly",
                "The system is responsive",
            ],
        )

        issues, score = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert "vague" in messages or "metric" in messages or "performance" in messages

    def test_recognizes_testable_gherkin_format(self, analyzer):
        """Test that Gherkin format is recognized as testable."""
        requirement = Requirement(
            title="Login Feature",
            description="User authentication flow.",
            acceptance_criteria=[
                "GIVEN valid credentials WHEN user clicks login THEN session is created",
                "GIVEN invalid password WHEN user clicks login THEN error message appears",
            ],
        )

        issues, score = analyzer.analyze(requirement)

        # Should have high testability score for Gherkin format
        assert score >= 0.6
        # Should have few untestability issues
        untestable_issues = [
            i for i in issues
            if "untestable" in i.message.lower() or "subjective" in i.message.lower()
        ]
        assert len(untestable_issues) == 0

    def test_recognizes_numeric_constraints(self, analyzer):
        """Test that numeric constraints are recognized as testable."""
        requirement = Requirement(
            title="API Performance",
            description="API response time requirements.",
            acceptance_criteria=[
                "API response time must be less than 200ms",
                "System handles at least 1000 concurrent requests",
                "Error rate must be below 0.1%",
            ],
        )

        issues, score = analyzer.analyze(requirement)

        # Should have high testability score for numeric constraints
        assert score >= 0.6

    def test_detects_restatement_of_title(self, analyzer):
        """Test detection of acceptance criteria that just restate the title."""
        requirement = Requirement(
            title="User Login Feature",
            description="Allow users to log in.",
            acceptance_criteria=[
                "User can login",  # Just restates title
                "Login works",     # Just restates title
            ],
        )

        issues, score = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert "restate" in messages or "title" in messages

    def test_no_acceptance_criteria_gives_low_score(self, analyzer):
        """Test that missing acceptance criteria results in low testability."""
        requirement = Requirement(
            title="Feature",
            description="Some feature description.",
            acceptance_criteria=[],
        )

        issues, score = analyzer.analyze(requirement)

        assert score < 0.5

    def test_all_issues_have_correct_category(self, analyzer):
        """Test that all returned issues have TESTABILITY category."""
        requirement = Requirement(
            title="UI Feature",
            description="Make it user-friendly and intuitive.",
            acceptance_criteria=["It works nicely"],
        )

        issues, _ = analyzer.analyze(requirement)

        for issue in issues:
            assert issue.category == IssueCategory.TESTABILITY


# =============================================================================
# Risk Analyzer Tests
# =============================================================================


class TestRiskAnalyzer:
    """Tests for RiskAnalyzer."""

    @pytest.fixture
    def analyzer(self, settings):
        """Create risk analyzer with test settings."""
        return RiskAnalyzer(settings)

    def test_detects_security_domain(self, analyzer):
        """Test detection of security-related requirements."""
        requirement = Requirement(
            title="User Authentication",
            description="Implement secure user authentication with password hashing.",
            acceptance_criteria=["Users can login with credentials"],
        )

        issues, _ = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert "security" in messages or "authentication" in messages

    def test_detects_financial_domain(self, analyzer):
        """Test detection of financial/payment-related requirements."""
        requirement = Requirement(
            title="Payment Processing",
            description="Process credit card payments for subscriptions.",
            acceptance_criteria=["Payment is processed successfully"],
        )

        issues, _ = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert "financial" in messages or "payment" in messages or "pci" in messages.lower()

    def test_detects_data_sensitive_domain(self, analyzer):
        """Test detection of data privacy requirements."""
        requirement = Requirement(
            title="User Data Export",
            description="Export user PII for GDPR compliance.",
            acceptance_criteria=["User can download their personal data"],
        )

        issues, _ = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert "data" in messages or "privacy" in messages or "sensitive" in messages

    def test_detects_integration_complexity(self, analyzer):
        """Test detection of third-party integration risks."""
        requirement = Requirement(
            title="Third-Party Integration",
            description="Integrate with external API for data sync.",
            acceptance_criteria=["Data syncs with vendor system"],
        )

        issues, _ = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert "integration" in messages or "third" in messages or "external" in messages

    def test_detects_complexity_indicators(self, analyzer):
        """Test detection of complexity indicators."""
        requirement = Requirement(
            title="Real-time Processing",
            description="Process data in real-time across distributed systems.",
            acceptance_criteria=["Data processed with concurrent updates"],
        )

        issues, _ = analyzer.analyze(requirement)

        messages = " ".join(i.message.lower() for i in issues)
        assert (
            "real-time" in messages or
            "distributed" in messages or
            "complexity" in messages or
            "concurrent" in messages
        )

    def test_low_risk_requirement_high_score(self, analyzer):
        """Test that low-risk requirements get high scores."""
        requirement = Requirement(
            title="Update Button Color",
            description="Change the submit button color from blue to green.",
            acceptance_criteria=[
                "Button displays with hex color #00FF00",
                "Color change appears in all themes",
            ],
        )

        issues, score = analyzer.analyze(requirement)

        # Should have high score (low risk)
        assert score >= 0.7
        # Should have few or no warning-level risk signals
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert len(warnings) <= 1

    def test_multiple_risk_domains_detected(self, analyzer):
        """Test that multiple risk domains are detected."""
        requirement = Requirement(
            title="Secure Payment Migration",
            description=(
                "Migrate user payment credentials from legacy system. "
                "Handle PCI compliance and GDPR requirements."
            ),
            acceptance_criteria=["Payment data migrated securely"],
        )

        issues, score = analyzer.analyze(requirement)

        # Should detect multiple risk domains
        messages = " ".join(i.message.lower() for i in issues)
        # Should mention at least 2 different risk areas
        risk_areas = sum([
            "security" in messages or "authentication" in messages,
            "financial" in messages or "payment" in messages,
            "data" in messages or "privacy" in messages,
            "migration" in messages or "legacy" in messages,
        ])
        assert risk_areas >= 2
        assert score < 0.8  # Lower score due to multiple risks

    def test_all_issues_have_correct_category(self, analyzer):
        """Test that all returned issues have RISK category."""
        requirement = Requirement(
            title="Payment Integration",
            description="Process payments via third-party gateway.",
            acceptance_criteria=["Payments work"],
        )

        issues, _ = analyzer.analyze(requirement)

        for issue in issues:
            assert issue.category == IssueCategory.RISK


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestAnalyzerEdgeCases:
    """Tests for edge cases across all analyzers."""

    @pytest.fixture
    def all_analyzers(self, settings):
        """Create all analyzers."""
        return {
            "ambiguity": AmbiguityAnalyzer(settings),
            "completeness": CompletenessAnalyzer(settings),
            "testability": TestabilityAnalyzer(settings),
            "risk": RiskAnalyzer(settings),
        }

    def test_empty_title_handled(self, all_analyzers):
        """Test that minimal title doesn't crash analyzers."""
        requirement = Requirement(
            title="X",
            description="A minimal feature.",
            acceptance_criteria=["Works"],
        )

        for name, analyzer in all_analyzers.items():
            issues, score = analyzer.analyze(requirement)
            assert isinstance(score, float), f"{name} failed"
            assert 0.0 <= score <= 1.0, f"{name} score out of range"

    def test_unicode_characters_handled(self, all_analyzers):
        """Test that unicode characters are handled correctly."""
        requirement = Requirement(
            title="Internationalization \u2764",
            description="Support for \u4e2d\u6587 and \u65e5\u672c\u8a9e characters.",
            acceptance_criteria=["\u0422\u0435\u0441\u0442 passes"],
        )

        for name, analyzer in all_analyzers.items():
            issues, score = analyzer.analyze(requirement)
            assert isinstance(score, float), f"{name} failed with unicode"

    def test_very_long_text_handled(self, all_analyzers):
        """Test that very long text is handled without timeout."""
        long_text = "Feature description. " * 500
        requirement = Requirement(
            title="Long Feature",
            description=long_text,
            acceptance_criteria=["Works correctly"] * 20,
        )

        for name, analyzer in all_analyzers.items():
            issues, score = analyzer.analyze(requirement)
            assert isinstance(score, float), f"{name} failed with long text"
            assert 0.0 <= score <= 1.0

    def test_special_regex_characters_handled(self, all_analyzers):
        """Test that regex special characters don't cause issues."""
        requirement = Requirement(
            title="Pattern Matching [v2.0]",
            description="Match patterns like (a|b)* and \\d+",
            acceptance_criteria=["Regex ^test$ works"],
        )

        for name, analyzer in all_analyzers.items():
            issues, score = analyzer.analyze(requirement)
            assert isinstance(score, float), f"{name} failed with regex chars"

    def test_all_analyzers_return_valid_issues(self, all_analyzers):
        """Test that all analyzers return valid Issue objects."""
        requirement = Requirement(
            title="Test Feature",
            description="A feature that should be tested appropriately.",
            acceptance_criteria=["Works nicely"],
        )

        for name, analyzer in all_analyzers.items():
            issues, _ = analyzer.analyze(requirement)
            for issue in issues:
                assert issue.severity in Severity, f"{name} invalid severity"
                assert issue.category in IssueCategory, f"{name} invalid category"
                assert issue.message, f"{name} empty message"
                assert issue.location, f"{name} empty location"


class TestAnalyzerScoring:
    """Tests for analyzer scoring consistency."""

    @pytest.fixture
    def all_analyzers(self, settings):
        """Create all analyzers."""
        return {
            "ambiguity": AmbiguityAnalyzer(settings),
            "completeness": CompletenessAnalyzer(settings),
            "testability": TestabilityAnalyzer(settings),
            "risk": RiskAnalyzer(settings),
        }

    def test_scores_always_in_valid_range(self, all_analyzers):
        """Test that scores are always between 0 and 1."""
        test_cases = [
            # Minimal requirement
            Requirement(title="X", description="", acceptance_criteria=[]),
            # Good requirement
            Requirement(
                title="User Login",
                description="Users can log in with email and password.",
                acceptance_criteria=["GIVEN valid creds WHEN login THEN success"],
            ),
            # Bad requirement
            Requirement(
                title="Feature",
                description="Do things appropriately etc.",
                acceptance_criteria=["Works"],
            ),
        ]

        for req in test_cases:
            for name, analyzer in all_analyzers.items():
                _, score = analyzer.analyze(req)
                assert 0.0 <= score <= 1.0, (
                    f"{name} score {score} out of range for {req.title}"
                )

    def test_worse_requirements_have_lower_scores(self, all_analyzers):
        """Test that worse requirements generally score lower."""
        good_req = Requirement(
            title="User Authentication",
            description=(
                "Users authenticate using email and password. "
                "The system validates credentials against the database."
            ),
            acceptance_criteria=[
                "GIVEN valid email and password WHEN user logs in THEN session created",
                "GIVEN invalid password WHEN user logs in THEN error 401 returned",
                "GIVEN locked account WHEN user logs in THEN error message shown",
            ],
        )

        bad_req = Requirement(
            title="Login",
            description="Users should login appropriately.",
            acceptance_criteria=["Works"],
        )

        for name, analyzer in all_analyzers.items():
            _, good_score = analyzer.analyze(good_req)
            _, bad_score = analyzer.analyze(bad_req)
            # Good requirement should generally score higher or equal
            # (Risk analyzer might flag security concerns in good req)
            if name != "risk":
                assert good_score >= bad_score - 0.1, (
                    f"{name}: good={good_score:.2f}, bad={bad_score:.2f}"
                )
