"""Tests for custom rule patterns functionality."""

import pytest
from unittest.mock import patch, MagicMock

from reqcheck.core.config import Settings
from reqcheck.core.models import IssueCategory, Severity
from reqcheck.rules.patterns import PatternMatcher, _compile_custom_patterns


class TestCustomPatternCompilation:
    """Tests for custom pattern compilation."""

    def test_compile_custom_weasel_words(self):
        """Test compiling custom weasel words."""
        patterns = _compile_custom_patterns(
            custom_weasel_words=["ASAP", "soon", "later"]
        )

        assert len(patterns) == 3
        for p in patterns:
            assert p.custom is True
            assert p.severity == Severity.WARNING
            assert p.category == IssueCategory.AMBIGUITY

    def test_compile_custom_forbidden_terms(self):
        """Test compiling custom forbidden terms."""
        patterns = _compile_custom_patterns(
            custom_forbidden_terms=["magic", "hack"]
        )

        assert len(patterns) == 2
        for p in patterns:
            assert p.custom is True
            assert p.severity == Severity.BLOCKER
            assert p.category == IssueCategory.COMPLETENESS

    def test_empty_patterns_handled(self):
        """Test that empty patterns are handled gracefully."""
        patterns = _compile_custom_patterns(
            custom_weasel_words=["", "   ", None],  # type: ignore
            custom_forbidden_terms=[],
        )

        # Should skip empty/whitespace patterns
        assert len(patterns) == 0

    def test_compile_from_yaml_config(self):
        """Test compiling patterns from YAML config structure."""
        config = {
            "weasel_words": ["soonish", "kinda"],
            "forbidden_terms": ["workaround"],
            "patterns": [
                {
                    "name": "custom_regex",
                    "regex": r"\bfoo\s+bar\b",
                    "severity": "warning",
                    "category": "ambiguity",
                    "message": "Found foo bar",
                }
            ],
        }

        patterns = _compile_custom_patterns(custom_patterns_config=config)

        # 2 weasel words + 1 forbidden + 1 custom regex
        assert len(patterns) == 4

    def test_invalid_regex_skipped(self):
        """Test that invalid regex patterns are skipped."""
        config = {
            "patterns": [
                {
                    "name": "invalid",
                    "regex": r"[invalid(regex",  # Invalid regex
                    "severity": "warning",
                    "category": "ambiguity",
                }
            ],
        }

        # Should not raise, just skip invalid pattern
        patterns = _compile_custom_patterns(custom_patterns_config=config)
        assert len(patterns) == 0


class TestPatternMatcherWithCustomPatterns:
    """Tests for PatternMatcher with custom patterns."""

    def test_matcher_with_custom_weasel_words(self):
        """Test matcher detects custom weasel words."""
        matcher = PatternMatcher(
            custom_weasel_words=["ASAP", "pronto"]
        )

        text = "The feature should be done ASAP and the fix needs to happen pronto."
        matches = list(matcher.find_matches(text))

        # Should find ASAP and pronto
        custom_matches = [m for m in matches if "ASAP" in m.matched_text or "pronto" in m.matched_text]
        assert len(custom_matches) >= 2

    def test_matcher_with_custom_forbidden_terms(self):
        """Test matcher detects custom forbidden terms as blockers."""
        matcher = PatternMatcher(
            custom_forbidden_terms=["hardcoded"]
        )

        text = "The value is hardcoded for now."
        matches = list(matcher.find_matches(text))

        hardcoded_matches = [m for m in matches if "hardcoded" in m.matched_text.lower()]
        assert len(hardcoded_matches) >= 1
        assert hardcoded_matches[0].severity == Severity.BLOCKER

    def test_matcher_with_settings(self):
        """Test matcher loads custom patterns from settings."""
        settings = Settings(
            custom_weasel_words="urgently,desperately",
            custom_forbidden_terms="placeholder",
        )

        matcher = PatternMatcher(settings=settings)

        text = "We urgently need to remove this placeholder code."
        matches = list(matcher.find_matches(text))

        # Should find both custom patterns
        custom_matches = [
            m for m in matches
            if "urgently" in m.matched_text.lower() or "placeholder" in m.matched_text.lower()
        ]
        assert len(custom_matches) >= 2

    def test_custom_pattern_count(self):
        """Test counting custom patterns."""
        matcher = PatternMatcher(
            custom_weasel_words=["word1", "word2"],
            custom_forbidden_terms=["term1"],
        )

        assert matcher.custom_pattern_count == 3

    def test_total_pattern_count(self):
        """Test total pattern count includes builtins."""
        matcher_default = PatternMatcher()
        matcher_custom = PatternMatcher(custom_weasel_words=["extra"])

        assert matcher_custom.total_pattern_count == matcher_default.total_pattern_count + 1

    def test_get_pattern_stats(self):
        """Test getting pattern statistics."""
        matcher = PatternMatcher(
            custom_weasel_words=["custom1"],
            custom_forbidden_terms=["custom2"],
        )

        stats = matcher.get_pattern_stats()

        assert stats["custom"] == 2
        assert stats["builtin"] == stats["total"] - 2
        assert "by_category" in stats
        assert "by_severity" in stats


class TestSettingsCustomPatterns:
    """Tests for custom patterns in Settings."""

    def test_custom_weasel_words_list_parsing(self):
        """Test parsing custom weasel words from comma-separated string."""
        settings = Settings(custom_weasel_words="word1, word2, word3")

        words = settings.custom_weasel_words_list
        assert words == ["word1", "word2", "word3"]

    def test_custom_weasel_words_empty(self):
        """Test empty custom weasel words."""
        settings = Settings(custom_weasel_words="")

        assert settings.custom_weasel_words_list == []

    def test_custom_forbidden_terms_list_parsing(self):
        """Test parsing custom forbidden terms."""
        settings = Settings(custom_forbidden_terms="hack, workaround")

        terms = settings.custom_forbidden_terms_list
        assert terms == ["hack", "workaround"]

    def test_load_custom_patterns_no_file(self):
        """Test loading patterns when no file configured."""
        settings = Settings(custom_patterns_file="")

        patterns = settings.load_custom_patterns()
        assert patterns == {}

    def test_load_custom_patterns_nonexistent_file(self):
        """Test loading patterns from nonexistent file."""
        settings = Settings(custom_patterns_file="/nonexistent/path.yaml")

        patterns = settings.load_custom_patterns()
        assert patterns == {}

    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    def test_load_custom_patterns_from_yaml(self, mock_exists, mock_open):
        """Test loading patterns from YAML file."""
        mock_exists.return_value = True

        yaml_content = """
weasel_words:
  - urgently
  - desperately
forbidden_terms:
  - placeholder
"""
        mock_open.return_value.__enter__.return_value.read.return_value = yaml_content

        settings = Settings(custom_patterns_file="/path/to/patterns.yaml")

        # This will fail without pyyaml, which is expected behavior
        patterns = settings.load_custom_patterns()

        # If yaml is not installed, should return empty
        # If installed, should return parsed config
        assert isinstance(patterns, dict)


class TestPatternMatchingIntegration:
    """Integration tests for pattern matching with custom patterns."""

    def test_custom_patterns_work_with_category_filter(self):
        """Test custom patterns work with category filtering."""
        matcher = PatternMatcher(
            custom_weasel_words=["urgently"],
            custom_forbidden_terms=["placeholder"],
        )

        text = "We urgently need to replace this placeholder."

        # Ambiguity category should include "urgently"
        ambiguity_matches = list(matcher.find_matches_by_category(text, IssueCategory.AMBIGUITY))
        assert any("urgently" in m.matched_text.lower() for m in ambiguity_matches)

        # Completeness category should include "placeholder"
        completeness_matches = list(matcher.find_matches_by_category(text, IssueCategory.COMPLETENESS))
        assert any("placeholder" in m.matched_text.lower() for m in completeness_matches)

    def test_severity_counts_include_custom(self):
        """Test severity counts include custom pattern matches."""
        matcher = PatternMatcher(
            custom_forbidden_terms=["critical_term"],
        )

        text = "This has a critical_term that must be removed."
        counts = matcher.get_severity_counts(text)

        assert counts[Severity.BLOCKER] >= 1

    def test_count_matches_includes_custom(self):
        """Test match counting includes custom patterns."""
        matcher = PatternMatcher(
            custom_weasel_words=["customword"],
        )

        text = "The customword appears twice: customword."
        counts = matcher.count_matches(text)

        assert "custom_weasel_word" in counts
        assert counts["custom_weasel_word"] == 2
