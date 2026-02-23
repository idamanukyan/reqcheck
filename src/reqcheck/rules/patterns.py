"""Rule-based pattern detection for requirements analysis."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

from reqcheck.core.models import IssueCategory, PatternMatch, Severity

if TYPE_CHECKING:
    from reqcheck.core.config import Settings


@dataclass
class Pattern:
    """Definition of a detection pattern."""

    name: str
    regex: re.Pattern
    severity: Severity
    category: IssueCategory
    message_template: str
    custom: bool = False  # Whether this is a custom user-defined pattern


# Weasel words that indicate vague requirements
WEASEL_WORDS = [
    r"\bappropriate(?:ly)?\b",
    r"\bproper(?:ly)?\b",
    r"\bcorrect(?:ly)?\b",
    r"\bsuitable\b",
    r"\bsufficient(?:ly)?\b",
    r"\badequate(?:ly)?\b",
    r"\breasonable\b",
    r"\befficient(?:ly)?\b",
    r"\beffective(?:ly)?\b",
    r"\boptimal(?:ly)?\b",
    r"\bas needed\b",
    r"\bas required\b",
    r"\bas appropriate\b",
    r"\bif necessary\b",
    r"\bwhen needed\b",
    r"\bin a timely manner\b",
    r"\buser[- ]friendly\b",
    r"\bintuitive(?:ly)?\b",
    r"\bseamless(?:ly)?\b",
    r"\brobust(?:ly)?\b",
    r"\bscalable\b",
    r"\bflexible\b",
    r"\bmaintainable\b",
]

# Vague quantifiers
VAGUE_QUANTIFIERS = [
    r"\bsome\b",
    r"\bmany\b",
    r"\bfew\b",
    r"\bseveral\b",
    r"\bvarious\b",
    r"\bmultiple\b",
    r"\bnumerous\b",
    r"\bmost\b",
    r"\bminimal\b",
    r"\bsignificant(?:ly)?\b",
    r"\blarge(?:ly)?\b",
    r"\bsmall\b",
    r"\bhigh(?:ly)?\b",
    r"\blow\b",
    r"\bfast(?:er)?\b",
    r"\bslow(?:er)?\b",
    r"\bquick(?:ly)?\b",
    r"\beasy\b",
    r"\beasily\b",
    r"\bsimple\b",
    r"\bsimply\b",
]

# Escape hatches and open-ended language
ESCAPE_HATCHES = [
    r"\betc\.?\b",
    r"\band so on\b",
    r"\band more\b",
    r"\band others?\b",
    r"\bsuch as\b(?![^.]*\))",  # "such as" not followed by closing paren
    r"\bfor example\b",
    r"\bincluding but not limited to\b",
    r"\bsimilar\b",
    r"\blike\b(?!\s+to\b)",  # "like" but not "like to"
    r"\b(?:and|or) similar\b",
    r"\bTBD\b",
    r"\bTBA\b",
    r"\bTBC\b",
    r"\bto be (?:determined|announced|confirmed)\b",
]

# Passive voice patterns (actor hidden)
PASSIVE_VOICE = [
    r"\bwill be (?:processed|validated|verified|checked|updated|created|deleted|modified|sent|received|stored|loaded|displayed|rendered|calculated|handled|managed)\b",
    r"\bis (?:processed|validated|verified|checked|updated|created|deleted|modified|sent|received|stored|loaded|displayed|rendered|calculated|handled|managed)\b",
    r"\bare (?:processed|validated|verified|checked|updated|created|deleted|modified|sent|received|stored|loaded|displayed|rendered|calculated|handled|managed)\b",
    r"\bshould be (?:processed|validated|verified|checked|updated|created|deleted|modified|sent|received|stored|loaded|displayed|rendered|calculated|handled|managed)\b",
    r"\bmust be (?:processed|validated|verified|checked|updated|created|deleted|modified|sent|received|stored|loaded|displayed|rendered|calculated|handled|managed)\b",
    r"\bcan be (?:processed|validated|verified|checked|updated|created|deleted|modified|sent|received|stored|loaded|displayed|rendered|calculated|handled|managed)\b",
]

# Ambiguous pronouns without clear antecedent
AMBIGUOUS_PRONOUNS = [
    r"\bit\s+(?:should|will|must|can|is|has|does)\b",
    r"\bthis\s+(?:should|will|must|can|is|has|does)\b",
    r"\bthat\s+(?:should|will|must|can|is|has|does)\b",
    r"\bthey\s+(?:should|will|must|can|are|have|do)\b",
    r"\bthe system\b",
    r"\bthe application\b",
    r"\bthe user\b(?!s?\s+(?:can|should|will|must))",  # "the user" without action verb
]

# Missing boundary/constraint indicators
MISSING_BOUNDARIES = [
    r"\bany\s+(?:file|data|input|value|user|request)\b",
    r"\ball\s+(?:files?|data|inputs?|values?|users?|requests?)\b",
    r"\bunlimited\b",
    r"\bno limit\b",
    r"\bwithout (?:limit|restriction)\b",
]

# Temporal ambiguity
TEMPORAL_AMBIGUITY = [
    r"\bimmediately\b",
    r"\bas soon as possible\b",
    r"\bASAP\b",
    r"\bquickly\b",
    r"\beventually\b",
    r"\blater\b",
    r"\bsoon\b",
    r"\bin real[- ]?time\b",
    r"\bperiodically\b",
    r"\bregularly\b",
    r"\bfrequently\b",
    r"\boccasionally\b",
]

# Weak modal verbs
WEAK_MODALS = [
    r"\bshould\b",
    r"\bcould\b",
    r"\bmight\b",
    r"\bmay\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bideally\b",
    r"\bpreferably\b",
    r"\boptionally\b",
]

# Risk signals
RISK_SIGNALS = [
    r"\bsecurity\b",
    r"\bauthenticat(?:ion|e)\b",
    r"\bauthoriz(?:ation|e)\b",
    r"\bpassword\b",
    r"\bcredential\b",
    r"\btoken\b",
    r"\bpayment\b",
    r"\btransaction\b",
    r"\bfinancial\b",
    r"\bmoney\b",
    r"\bcredit card\b",
    r"\bPII\b",
    r"\bpersonal(?:ly identifiable)?\s+(?:information|data)\b",
    r"\bGDPR\b",
    r"\bcompliance\b",
    r"\bregulat(?:ory|ion)\b",
    r"\bmigrat(?:ion|e)\b",
    r"\blegacy\b",
    r"\bthird[- ]?party\b",
    r"\bexternal\s+(?:API|service|system)\b",
    r"\bintegrat(?:ion|e)\b",
]


def _compile_patterns() -> list[Pattern]:
    """Compile all pattern definitions."""
    patterns = []

    # Weasel words
    for regex_str in WEASEL_WORDS:
        patterns.append(
            Pattern(
                name="weasel_word",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.WARNING,
                category=IssueCategory.AMBIGUITY,
                message_template='Vague term "{text}" - specify measurable criteria',
            )
        )

    # Vague quantifiers
    for regex_str in VAGUE_QUANTIFIERS:
        patterns.append(
            Pattern(
                name="vague_quantifier",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.WARNING,
                category=IssueCategory.AMBIGUITY,
                message_template='Vague quantifier "{text}" - specify exact values or ranges',
            )
        )

    # Escape hatches
    for regex_str in ESCAPE_HATCHES:
        patterns.append(
            Pattern(
                name="escape_hatch",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.BLOCKER,
                category=IssueCategory.COMPLETENESS,
                message_template='Open-ended language "{text}" - enumerate all cases explicitly',
            )
        )

    # Passive voice
    for regex_str in PASSIVE_VOICE:
        patterns.append(
            Pattern(
                name="passive_voice",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.WARNING,
                category=IssueCategory.AMBIGUITY,
                message_template='Passive voice "{text}" - specify who/what performs this action',
            )
        )

    # Ambiguous pronouns
    for regex_str in AMBIGUOUS_PRONOUNS:
        patterns.append(
            Pattern(
                name="ambiguous_pronoun",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.WARNING,
                category=IssueCategory.AMBIGUITY,
                message_template='Ambiguous reference "{text}" - use specific noun instead',
            )
        )

    # Missing boundaries
    for regex_str in MISSING_BOUNDARIES:
        patterns.append(
            Pattern(
                name="missing_boundary",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.WARNING,
                category=IssueCategory.COMPLETENESS,
                message_template='Unbounded scope "{text}" - define limits and constraints',
            )
        )

    # Temporal ambiguity
    for regex_str in TEMPORAL_AMBIGUITY:
        patterns.append(
            Pattern(
                name="temporal_ambiguity",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.WARNING,
                category=IssueCategory.AMBIGUITY,
                message_template='Vague timing "{text}" - specify exact duration or deadline',
            )
        )

    # Weak modals
    for regex_str in WEAK_MODALS:
        patterns.append(
            Pattern(
                name="weak_modal",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.SUGGESTION,
                category=IssueCategory.AMBIGUITY,
                message_template='Weak requirement "{text}" - use "must" or "will" for mandatory features',
            )
        )

    # Risk signals
    for regex_str in RISK_SIGNALS:
        patterns.append(
            Pattern(
                name="risk_signal",
                regex=re.compile(regex_str, re.IGNORECASE),
                severity=Severity.SUGGESTION,
                category=IssueCategory.RISK,
                message_template='Risk area detected "{text}" - ensure proper review and security assessment',
            )
        )

    return patterns


def _compile_custom_patterns(
    custom_weasel_words: list[str] | None = None,
    custom_forbidden_terms: list[str] | None = None,
    custom_patterns_config: dict[str, Any] | None = None,
) -> list[Pattern]:
    """Compile custom user-defined patterns.

    Args:
        custom_weasel_words: List of additional weasel words
        custom_forbidden_terms: List of forbidden terms (blockers)
        custom_patterns_config: Full custom patterns from YAML file

    Returns:
        List of compiled custom patterns
    """
    patterns = []

    # Add custom weasel words
    if custom_weasel_words:
        for word in custom_weasel_words:
            if word and word.strip():
                # Escape special regex characters and wrap in word boundaries
                escaped = re.escape(word.strip())
                patterns.append(
                    Pattern(
                        name="custom_weasel_word",
                        regex=re.compile(rf"\b{escaped}\b", re.IGNORECASE),
                        severity=Severity.WARNING,
                        category=IssueCategory.AMBIGUITY,
                        message_template=f'Custom vague term "{{text}}" - specify measurable criteria',
                        custom=True,
                    )
                )

    # Add custom forbidden terms (treated as blockers)
    if custom_forbidden_terms:
        for term in custom_forbidden_terms:
            if term:
                escaped = re.escape(term)
                patterns.append(
                    Pattern(
                        name="custom_forbidden_term",
                        regex=re.compile(rf"\b{escaped}\b", re.IGNORECASE),
                        severity=Severity.BLOCKER,
                        category=IssueCategory.COMPLETENESS,
                        message_template=f'Forbidden term "{{text}}" - this term is not allowed',
                        custom=True,
                    )
                )

    # Process custom patterns from YAML config
    if custom_patterns_config:
        # Custom weasel words from YAML
        yaml_weasel = custom_patterns_config.get("weasel_words", [])
        for word in yaml_weasel:
            if word and word not in (custom_weasel_words or []):
                escaped = re.escape(word)
                patterns.append(
                    Pattern(
                        name="custom_weasel_word",
                        regex=re.compile(rf"\b{escaped}\b", re.IGNORECASE),
                        severity=Severity.WARNING,
                        category=IssueCategory.AMBIGUITY,
                        message_template=f'Custom vague term "{{text}}" - specify measurable criteria',
                        custom=True,
                    )
                )

        # Custom forbidden terms from YAML
        yaml_forbidden = custom_patterns_config.get("forbidden_terms", [])
        for term in yaml_forbidden:
            if term and term not in (custom_forbidden_terms or []):
                escaped = re.escape(term)
                patterns.append(
                    Pattern(
                        name="custom_forbidden_term",
                        regex=re.compile(rf"\b{escaped}\b", re.IGNORECASE),
                        severity=Severity.BLOCKER,
                        category=IssueCategory.COMPLETENESS,
                        message_template=f'Forbidden term "{{text}}" - this term is not allowed',
                        custom=True,
                    )
                )

        # Custom regex patterns from YAML
        yaml_patterns = custom_patterns_config.get("patterns", [])
        for pattern_def in yaml_patterns:
            if isinstance(pattern_def, dict):
                try:
                    regex_str = pattern_def.get("regex", "")
                    if not regex_str:
                        continue

                    severity_str = pattern_def.get("severity", "warning").lower()
                    severity = Severity(severity_str) if severity_str in [s.value for s in Severity] else Severity.WARNING

                    category_str = pattern_def.get("category", "ambiguity").lower()
                    category = IssueCategory(category_str) if category_str in [c.value for c in IssueCategory] else IssueCategory.AMBIGUITY

                    patterns.append(
                        Pattern(
                            name=pattern_def.get("name", "custom_pattern"),
                            regex=re.compile(regex_str, re.IGNORECASE),
                            severity=severity,
                            category=category,
                            message_template=pattern_def.get("message", 'Pattern match "{text}"'),
                            custom=True,
                        )
                    )
                except (re.error, ValueError):
                    # Skip invalid patterns
                    continue

    return patterns


class PatternMatcher:
    """Rule-based pattern matcher for requirements text."""

    def __init__(
        self,
        settings: "Settings | None" = None,
        custom_weasel_words: list[str] | None = None,
        custom_forbidden_terms: list[str] | None = None,
    ):
        """Initialize pattern matcher with optional custom patterns.

        Args:
            settings: Application settings (for loading custom patterns)
            custom_weasel_words: Additional weasel words to detect
            custom_forbidden_terms: Forbidden terms that cause blockers
        """
        self._patterns = _compile_patterns()

        # Collect custom patterns from various sources
        all_custom_weasel = list(custom_weasel_words or [])
        all_custom_forbidden = list(custom_forbidden_terms or [])
        custom_config: dict[str, Any] = {}

        if settings:
            all_custom_weasel.extend(settings.custom_weasel_words_list)
            all_custom_forbidden.extend(settings.custom_forbidden_terms_list)
            custom_config = settings.load_custom_patterns()

        # Compile and add custom patterns
        custom_patterns = _compile_custom_patterns(
            custom_weasel_words=all_custom_weasel,
            custom_forbidden_terms=all_custom_forbidden,
            custom_patterns_config=custom_config,
        )
        self._patterns.extend(custom_patterns)
        self._custom_pattern_count = len(custom_patterns)

    def find_matches(self, text: str) -> Iterator[PatternMatch]:
        """Find all pattern matches in the given text."""
        for pattern in self._patterns:
            for match in pattern.regex.finditer(text):
                yield PatternMatch(
                    pattern_name=pattern.name,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=pattern.severity,
                    category=pattern.category,
                    message_template=pattern.message_template,
                )

    def find_matches_by_category(
        self, text: str, category: IssueCategory
    ) -> Iterator[PatternMatch]:
        """Find matches for a specific category."""
        for match in self.find_matches(text):
            if match.category == category:
                yield match

    def count_matches(self, text: str) -> dict[str, int]:
        """Count matches by pattern name."""
        counts: dict[str, int] = {}
        for match in self.find_matches(text):
            counts[match.pattern_name] = counts.get(match.pattern_name, 0) + 1
        return counts

    def get_severity_counts(self, text: str) -> dict[Severity, int]:
        """Count matches by severity."""
        counts: dict[Severity, int] = {s: 0 for s in Severity}
        for match in self.find_matches(text):
            counts[match.severity] += 1
        return counts

    @property
    def custom_pattern_count(self) -> int:
        """Get the number of custom patterns loaded."""
        return self._custom_pattern_count

    @property
    def total_pattern_count(self) -> int:
        """Get the total number of patterns."""
        return len(self._patterns)

    def get_pattern_stats(self) -> dict[str, int]:
        """Get statistics about loaded patterns."""
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        custom_count = 0

        for pattern in self._patterns:
            cat_name = pattern.category.value
            by_category[cat_name] = by_category.get(cat_name, 0) + 1

            sev_name = pattern.severity.value
            by_severity[sev_name] = by_severity.get(sev_name, 0) + 1

            if pattern.custom:
                custom_count += 1

        return {
            "total": len(self._patterns),
            "custom": custom_count,
            "builtin": len(self._patterns) - custom_count,
            "by_category": by_category,
            "by_severity": by_severity,
        }
