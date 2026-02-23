"""Async analyzer implementations for parallel execution."""

import re
from abc import ABC, abstractmethod
from typing import Any

from reqcheck.core.config import Settings, get_settings
from reqcheck.core.constants import (
    BONUS_TESTABLE_PATTERNS,
    MAX_EVIDENCE_MATCHES,
    MIN_ACCEPTANCE_CRITERIA_COUNT,
    MIN_ACCEPTANCE_CRITERION_LENGTH,
    MIN_DESCRIPTION_LENGTH,
    PENALTY_MISSING_ACCEPTANCE_CRITERIA,
    PENALTY_MULTIPLE_RISK_FACTORS,
    PENALTY_REDUCTION_FACTOR_LONG_TEXT,
    PENALTY_SHORT_DESCRIPTION,
    RESTATEMENT_OVERLAP_THRESHOLD,
    RISK_FACTORS_HIGH_THRESHOLD,
    SCORE_BASELINE_NO_LLM,
    SCORE_BASELINE_TESTABILITY,
    SCORE_DEFAULT_LLM_FALLBACK,
    SCORE_NO_ACCEPTANCE_CRITERIA,
    SCORE_PERFECT,
    SEVERITY_WEIGHT_DEFAULT,
    SHORT_CRITERION_THRESHOLD,
    TEXT_LENGTH_LONG_THRESHOLD,
    get_completeness_severity_weights,
    get_risk_severity_weights,
    get_severity_weights,
)
from reqcheck.core.exceptions import LLMClientError
from reqcheck.core.logging import get_logger
from reqcheck.core.models import Issue, IssueCategory, Requirement, Severity
from reqcheck.llm.async_client import AsyncLLMClient
from reqcheck.rules.patterns import PatternMatcher

logger = get_logger("analyzers.async")


class AsyncBaseAnalyzer(ABC):
    """Abstract base class for async requirement analyzers."""

    category: IssueCategory

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: AsyncLLMClient | None = None,
        pattern_matcher: PatternMatcher | None = None,
    ):
        self._settings = settings or get_settings()
        self._llm_client = llm_client
        self._pattern_matcher = pattern_matcher or PatternMatcher()

    @property
    def llm_client(self) -> AsyncLLMClient:
        """Lazy-initialize async LLM client."""
        if self._llm_client is None:
            self._llm_client = AsyncLLMClient(self._settings)
        return self._llm_client

    @abstractmethod
    async def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """Analyze a requirement and return issues with a score."""
        pass

    def _run_rule_based_analysis(self, requirement: Requirement) -> list[Issue]:
        """Run rule-based pattern matching for this category."""
        issues = []
        text = requirement.full_text

        for match in self._pattern_matcher.find_matches_by_category(text, self.category):
            location = self._determine_location(requirement, match.start_pos)
            issue = match.to_issue(location)
            issues.append(issue)

        return issues

    def _determine_location(self, requirement: Requirement, position: int) -> str:
        """Determine which field a position falls into."""
        title_len = len(requirement.title)
        desc_start = title_len + 1
        desc_end = desc_start + len(requirement.description)

        if position < title_len:
            return "title"
        elif position < desc_end:
            return "description"
        else:
            ac_text = "\n".join(requirement.acceptance_criteria)
            remaining_pos = position - desc_end - len("Acceptance Criteria:\n")
            current_pos = 0
            for i, ac in enumerate(requirement.acceptance_criteria):
                ac_len = len(f"- {ac}\n")
                if current_pos + ac_len > remaining_pos:
                    return f"acceptance_criteria[{i}]"
                current_pos += ac_len
            return "acceptance_criteria"

    def _parse_llm_issues(
        self, llm_response: dict[str, Any], default_category: IssueCategory
    ) -> list[Issue]:
        """Parse issues from LLM response."""
        issues = []
        for issue_data in llm_response.get("issues", []):
            try:
                severity_str = issue_data.get("severity", "warning").lower()
                severity = (
                    Severity(severity_str)
                    if severity_str in [s.value for s in Severity]
                    else Severity.WARNING
                )

                issues.append(
                    Issue(
                        severity=severity,
                        category=default_category,
                        location=issue_data.get("location", "unknown"),
                        message=issue_data.get("message", ""),
                        suggestion=issue_data.get("suggestion", ""),
                        evidence=issue_data.get("evidence", ""),
                    )
                )
            except (KeyError, ValueError):
                continue

        return issues

    def _deduplicate_issues(self, issues: list[Issue]) -> list[Issue]:
        """Remove duplicate issues based on message similarity."""
        seen_messages: set[str] = set()
        unique_issues = []

        for issue in issues:
            normalized = issue.message.lower().strip()
            if normalized not in seen_messages:
                seen_messages.add(normalized)
                unique_issues.append(issue)

        return unique_issues

    def _merge_issues(
        self, rule_issues: list[Issue], llm_issues: list[Issue]
    ) -> list[Issue]:
        """Merge rule-based and LLM issues, removing duplicates."""
        all_issues = llm_issues + rule_issues
        return self._deduplicate_issues(all_issues)


class AsyncAmbiguityAnalyzer(AsyncBaseAnalyzer):
    """Async analyzer for detecting ambiguous language in requirements."""

    category = IssueCategory.AMBIGUITY

    async def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """Analyze requirement for ambiguity issues."""
        rule_issues: list[Issue] = []
        llm_issues: list[Issue] = []
        score = SCORE_PERFECT

        # Run rule-based analysis
        if self._settings.enable_rule_based_analysis:
            rule_issues = self._run_rule_based_analysis(requirement)
            logger.debug(
                "Rule-based analysis complete",
                extra={"issue_count": len(rule_issues)},
            )

        # Run LLM analysis
        if self._settings.llm_available:
            try:
                response = await self.llm_client.analyze_ambiguity(requirement.full_text)
                llm_issues = self._parse_llm_issues(response, self.category)
                score = response.get("ambiguity_score", SCORE_DEFAULT_LLM_FALLBACK)
                logger.debug(
                    "LLM analysis complete",
                    extra={"issue_count": len(llm_issues), "score": score},
                )
            except LLMClientError as e:
                logger.warning(
                    "LLM analysis failed, using rule-based scoring",
                    extra={"error": str(e)},
                )
                score = self._estimate_score_from_rules(rule_issues, requirement)

        all_issues = self._merge_issues(rule_issues, llm_issues)

        if not self._settings.llm_available:
            score = self._estimate_score_from_rules(all_issues, requirement)

        return all_issues, score

    def _estimate_score_from_rules(
        self, issues: list[Issue], requirement: Requirement
    ) -> float:
        """Estimate ambiguity score based on rule matches."""
        if not issues:
            return SCORE_BASELINE_NO_LLM

        weights = get_severity_weights()
        penalty = sum(weights.get(i.severity.value, SEVERITY_WEIGHT_DEFAULT) for i in issues)

        text_len = len(requirement.full_text)
        if text_len > TEXT_LENGTH_LONG_THRESHOLD:
            penalty *= PENALTY_REDUCTION_FACTOR_LONG_TEXT

        return max(0.0, min(1.0, SCORE_PERFECT - penalty))


class AsyncCompletenessAnalyzer(AsyncBaseAnalyzer):
    """Async analyzer for detecting completeness issues in requirements."""

    category = IssueCategory.COMPLETENESS

    async def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """Analyze requirement for completeness issues."""
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
            logger.debug(
                "Rule-based analysis complete",
                extra={"issue_count": len(rule_issues)},
            )

        # Run LLM analysis
        if self._settings.llm_available:
            try:
                response = await self.llm_client.analyze_completeness(requirement.full_text)
                llm_issues = self._parse_llm_issues(response, self.category)
                score = response.get("completeness_score", SCORE_DEFAULT_LLM_FALLBACK)

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

                logger.debug(
                    "LLM analysis complete",
                    extra={"issue_count": len(llm_issues), "score": score},
                )
            except LLMClientError as e:
                logger.warning(
                    "LLM analysis failed, using rule-based scoring",
                    extra={"error": str(e)},
                )
                score = self._estimate_score_from_rules(rule_issues, requirement)

        all_issues = self._merge_issues(rule_issues, llm_issues)

        if not self._settings.llm_available:
            score = self._estimate_score_from_rules(all_issues, requirement)

        return all_issues, score

    def _check_structural_completeness(self, requirement: Requirement) -> list[Issue]:
        """Check for basic structural completeness."""
        issues = []

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

        if not requirement.acceptance_criteria:
            base_score -= PENALTY_MISSING_ACCEPTANCE_CRITERIA

        if len(requirement.description) < MIN_DESCRIPTION_LENGTH:
            base_score -= PENALTY_SHORT_DESCRIPTION

        weights = get_completeness_severity_weights()
        for issue in issues:
            base_score -= weights.get(issue.severity.value, SEVERITY_WEIGHT_DEFAULT)

        return max(0.0, min(1.0, base_score))


class AsyncTestabilityAnalyzer(AsyncBaseAnalyzer):
    """Async analyzer for evaluating testability of requirements."""

    category = IssueCategory.TESTABILITY

    TESTABLE_PATTERNS = [
        r"\bGIVEN\b.*\bWHEN\b.*\bTHEN\b",
        r"\bmust\s+(?:be|have|return|display|show)\b",
        r"\bshall\s+(?:be|have|return|display|show)\b",
        r"\bwill\s+(?:be|have|return|display|show)\b",
        r"\b\d+\s*(?:ms|seconds?|minutes?|hours?|days?)\b",
        r"\b\d+\s*(?:MB|GB|KB|bytes?)\b",
        r"\b(?:exactly|at least|at most|between)\s+\d+\b",
        r"\berror\s+(?:code|message)\b",
        r"\breturn(?:s|ed)?\s+(?:true|false|null|empty|\d+)\b",
    ]

    UNTESTABLE_PATTERNS = [
        (r"\b(?:works?|functions?)\s+(?:correctly|properly|well)\b", "Vague success criterion"),
        (r"\buser[- ]friendly\b", "Subjective quality"),
        (r"\bintuitive(?:ly)?\b", "Subjective quality"),
        (r"\bseamless(?:ly)?\b", "Subjective quality"),
        (r"\beasy\s+to\s+(?:use|understand|read)\b", "Subjective quality"),
        (r"\bnice\s+(?:UI|interface|experience)\b", "Subjective quality"),
        (r"\bgood\s+(?:performance|UX|experience)\b", "Subjective quality"),
        (r"\bfast(?:er)?\b(?!\s*\d)", "Vague performance without metric"),
        (r"\bslow(?:er)?\b(?!\s*\d)", "Vague performance without metric"),
        (r"\bquick(?:ly)?\b(?!\s*\d)", "Vague timing without metric"),
        (r"\bresponsive\b(?!\s*\d)", "Vague performance without metric"),
        (r"\bsecure(?:ly)?\b(?!\s+(?:using|with|via))", "Vague security without specifics"),
    ]

    async def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """Analyze requirement for testability issues."""
        rule_issues: list[Issue] = []
        llm_issues: list[Issue] = []
        score = SCORE_PERFECT

        ac_issues = self._check_acceptance_criteria_testability(requirement)
        rule_issues.extend(ac_issues)

        if self._settings.enable_rule_based_analysis:
            pattern_issues = self._run_rule_based_analysis(requirement)
            rule_issues.extend(pattern_issues)
            logger.debug(
                "Rule-based analysis complete",
                extra={"issue_count": len(rule_issues)},
            )

        if self._settings.llm_available:
            try:
                response = await self.llm_client.analyze_testability(requirement.full_text)
                llm_issues = self._parse_llm_issues(response, self.category)
                score = response.get("testability_score", SCORE_DEFAULT_LLM_FALLBACK)

                scenarios = response.get("suggested_test_scenarios", [])
                logger.debug(
                    "LLM analysis complete",
                    extra={
                        "issue_count": len(llm_issues),
                        "score": score,
                        "suggested_scenarios": len(scenarios),
                    },
                )
            except LLMClientError as e:
                logger.warning(
                    "LLM analysis failed, using rule-based scoring",
                    extra={"error": str(e)},
                )
                score = self._estimate_score_from_rules(rule_issues, requirement)

        all_issues = self._merge_issues(rule_issues, llm_issues)

        if not self._settings.llm_available:
            score = self._estimate_score_from_rules(all_issues, requirement)

        return all_issues, score

    def _check_acceptance_criteria_testability(
        self, requirement: Requirement
    ) -> list[Issue]:
        """Check each acceptance criterion for testability."""
        issues = []

        for i, ac in enumerate(requirement.acceptance_criteria):
            location = f"acceptance_criteria[{i}]"

            has_testable_pattern = any(
                re.search(pattern, ac, re.IGNORECASE)
                for pattern in self.TESTABLE_PATTERNS
            )

            for pattern, reason in self.UNTESTABLE_PATTERNS:
                match = re.search(pattern, ac, re.IGNORECASE)
                if match:
                    issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            category=self.category,
                            location=location,
                            message=f"Untestable criterion: {reason}",
                            suggestion="Replace with measurable, objective criteria",
                            evidence=match.group(),
                        )
                    )

            if not has_testable_pattern and len(ac) < SHORT_CRITERION_THRESHOLD:
                if self._is_restatement(ac, requirement.title):
                    issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            category=self.category,
                            location=location,
                            message=(
                                "Acceptance criterion appears to restate the title "
                                "without adding testable detail"
                            ),
                            suggestion="Add specific conditions, inputs, and expected outputs",
                            evidence=ac,
                        )
                    )

        return issues

    def _is_restatement(self, ac: str, title: str) -> bool:
        """Check if acceptance criterion is just restating the title."""
        ac_words = set(re.findall(r"\w+", ac.lower()))
        title_words = set(re.findall(r"\w+", title.lower()))

        stop_words = {
            "the", "a", "an", "is", "are", "will", "should",
            "must", "can", "be", "to", "and", "or", "it"
        }
        ac_words -= stop_words
        title_words -= stop_words

        if not ac_words or not title_words:
            return False

        overlap = len(ac_words & title_words) / len(ac_words)
        return overlap > RESTATEMENT_OVERLAP_THRESHOLD

    def _estimate_score_from_rules(
        self, issues: list[Issue], requirement: Requirement
    ) -> float:
        """Estimate testability score based on patterns and issues."""
        if not requirement.acceptance_criteria:
            return SCORE_NO_ACCEPTANCE_CRITERIA

        base_score = SCORE_BASELINE_TESTABILITY

        testable_count = 0
        for ac in requirement.acceptance_criteria:
            if any(
                re.search(pattern, ac, re.IGNORECASE)
                for pattern in self.TESTABLE_PATTERNS
            ):
                testable_count += 1

        if requirement.acceptance_criteria:
            testable_ratio = testable_count / len(requirement.acceptance_criteria)
            base_score += testable_ratio * BONUS_TESTABLE_PATTERNS

        weights = get_severity_weights()
        for issue in issues:
            base_score -= weights.get(issue.severity.value, SEVERITY_WEIGHT_DEFAULT)

        return max(0.0, min(1.0, base_score))


class AsyncRiskAnalyzer(AsyncBaseAnalyzer):
    """Async analyzer for identifying delivery and quality risks."""

    category = IssueCategory.RISK

    HIGH_RISK_DOMAINS = {
        "security": [
            r"\bauthenticat(?:ion|e|ing)\b",
            r"\bauthoriz(?:ation|e|ing)\b",
            r"\bpassword\b",
            r"\bcredential\b",
            r"\btoken\b",
            r"\bsession\b",
            r"\bencrypt(?:ion|ed)?\b",
            r"\bSSO\b",
            r"\bOAuth\b",
            r"\bJWT\b",
            r"\bACL\b",
            r"\bpermission\b",
            r"\baccess control\b",
        ],
        "financial": [
            r"\bpayment\b",
            r"\btransaction\b",
            r"\bbilling\b",
            r"\binvoice\b",
            r"\bsubscription\b",
            r"\brefund\b",
            r"\bcredit card\b",
            r"\bbank\b",
            r"\bmoney\b",
            r"\bcurrency\b",
            r"\bPCI\b",
        ],
        "data_sensitive": [
            r"\bPII\b",
            r"\bpersonal(?:ly identifiable)?\s+(?:information|data)\b",
            r"\bGDPR\b",
            r"\bCCPA\b",
            r"\bHIPAA\b",
            r"\bPHI\b",
            r"\bsensitive data\b",
            r"\buser data\b",
            r"\bprivacy\b",
        ],
        "integration": [
            r"\bthird[- ]?party\b",
            r"\bexternal\s+(?:API|service|system)\b",
            r"\bintegrat(?:ion|e|ing)\b",
            r"\bwebhook\b",
            r"\bAPI\s+(?:call|request|endpoint)\b",
            r"\bvendor\b",
            r"\blegacy\b",
        ],
        "data_migration": [
            r"\bmigrat(?:ion|e|ing)\b",
            r"\bimport(?:ing)?\b",
            r"\bexport(?:ing)?\b",
            r"\bconvert(?:ing|sion)?\b",
            r"\btransform(?:ation|ing)?\b",
            r"\bETL\b",
            r"\bdata\s+(?:transfer|move|copy)\b",
        ],
    }

    COMPLEXITY_INDICATORS = [
        (r"\bif\b.*\bthen\b.*\belse\b", "Multiple conditional branches"),
        (r"\b(?:and|or)\s+(?:if|when)\b", "Compound conditions"),
        (r"\bdepend(?:s|ing|ent)?\s+on\b", "External dependencies"),
        (r"\bacross\s+(?:multiple|all|different)\b", "Cross-cutting concern"),
        (r"\breal[- ]?time\b", "Real-time requirements"),
        (r"\bconcurren(?:t|cy)\b", "Concurrency handling"),
        (r"\bdistributed\b", "Distributed system"),
        (r"\bscale\b|\bscaling\b", "Scalability concern"),
        (r"\bbackward(?:s)?\s+compatib(?:le|ility)\b", "Backward compatibility"),
    ]

    async def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """Analyze requirement for risk signals."""
        rule_issues: list[Issue] = []
        llm_issues: list[Issue] = []
        risk_factors: list[str] = []

        domain_issues = self._identify_risk_domains(requirement)
        rule_issues.extend(domain_issues)

        complexity_issues = self._identify_complexity(requirement)
        rule_issues.extend(complexity_issues)

        if self._settings.enable_rule_based_analysis:
            pattern_issues = self._run_rule_based_analysis(requirement)
            rule_issues.extend(pattern_issues)
            logger.debug(
                "Rule-based analysis complete",
                extra={"signal_count": len(rule_issues)},
            )

        if self._settings.llm_available:
            try:
                response = await self.llm_client.analyze_risk(requirement.full_text)
                llm_issues = self._parse_llm_issues(response, self.category)
                risk_factors = response.get("risk_factors", [])

                for review in response.get("recommended_reviews", []):
                    llm_issues.append(
                        Issue(
                            severity=Severity.SUGGESTION,
                            category=self.category,
                            location="requirement",
                            message=f"Recommended: {review}",
                            suggestion=f"Schedule {review} before implementation",
                        )
                    )

                logger.debug(
                    "LLM analysis complete",
                    extra={
                        "issue_count": len(llm_issues),
                        "risk_factors": len(risk_factors),
                    },
                )
            except LLMClientError as e:
                logger.warning(
                    "LLM analysis failed",
                    extra={"error": str(e)},
                )

        all_issues = self._merge_issues(rule_issues, llm_issues)
        score = self._calculate_risk_score(all_issues, risk_factors)

        return all_issues, score

    def _identify_risk_domains(self, requirement: Requirement) -> list[Issue]:
        """Identify which high-risk domains this requirement touches."""
        issues = []
        text = requirement.full_text

        for domain, patterns in self.HIGH_RISK_DOMAINS.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)

            if matches:
                severity = self._get_domain_severity(domain)
                domain_name = domain.replace("_", " ")
                issues.append(
                    Issue(
                        severity=severity,
                        category=self.category,
                        location="requirement",
                        message=f"Touches {domain_name} domain - requires additional review",
                        suggestion=self._get_domain_suggestion(domain),
                        evidence=", ".join(list(set(matches))[:MAX_EVIDENCE_MATCHES]),
                    )
                )

        return issues

    def _get_domain_severity(self, domain: str) -> Severity:
        """Get severity level for a risk domain."""
        high_risk = {"security", "financial", "data_sensitive"}
        if domain in high_risk:
            return Severity.WARNING
        return Severity.SUGGESTION

    def _get_domain_suggestion(self, domain: str) -> str:
        """Get suggested action for a risk domain."""
        suggestions = {
            "security": "Ensure security review and threat modeling before implementation",
            "financial": "Verify PCI compliance requirements and audit logging",
            "data_sensitive": "Review data handling policies and privacy requirements",
            "integration": "Document API contracts and failure handling",
            "data_migration": "Plan rollback strategy and data validation steps",
        }
        return suggestions.get(domain, "Schedule appropriate review")

    def _identify_complexity(self, requirement: Requirement) -> list[Issue]:
        """Identify complexity indicators in the requirement."""
        issues = []
        text = requirement.full_text

        for pattern, description in self.COMPLEXITY_INDICATORS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                issues.append(
                    Issue(
                        severity=Severity.SUGGESTION,
                        category=self.category,
                        location="requirement",
                        message=f"Complexity indicator: {description}",
                        suggestion="Consider breaking down or adding more detail",
                        evidence=match.group(),
                    )
                )

        return issues

    def _calculate_risk_score(
        self, issues: list[Issue], risk_factors: list[str]
    ) -> float:
        """Calculate risk score (0.0 = highest risk, 1.0 = lowest risk)."""
        base_score = SCORE_PERFECT

        weights = get_risk_severity_weights()
        for issue in issues:
            base_score -= weights.get(issue.severity.value, SEVERITY_WEIGHT_DEFAULT)

        if len(risk_factors) > RISK_FACTORS_HIGH_THRESHOLD:
            base_score -= PENALTY_MULTIPLE_RISK_FACTORS

        return max(0.0, min(1.0, base_score))
