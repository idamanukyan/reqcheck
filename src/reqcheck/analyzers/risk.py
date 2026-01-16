"""Risk assessment analyzer."""

import logging
import re

from reqcheck.analyzers.base import BaseAnalyzer
from reqcheck.core.constants import (
    MAX_EVIDENCE_MATCHES,
    PENALTY_MULTIPLE_RISK_FACTORS,
    RISK_FACTORS_HIGH_THRESHOLD,
    SCORE_PERFECT,
    SEVERITY_WEIGHT_DEFAULT,
    get_risk_severity_weights,
)
from reqcheck.core.models import Issue, IssueCategory, Requirement, Severity
from reqcheck.llm.client import LLMClientError

logger = logging.getLogger(__name__)


class RiskAnalyzer(BaseAnalyzer):
    """Analyzer for identifying delivery and quality risks from requirements."""

    category = IssueCategory.RISK

    # High-risk domain indicators
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

    # Complexity indicators
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

    def analyze(self, requirement: Requirement) -> tuple[list[Issue], float]:
        """
        Analyze requirement for risk signals.

        Identifies:
        - Security risks
        - Financial/compliance risks
        - Integration risks
        - Complexity risks
        - Data risks
        """
        rule_issues: list[Issue] = []
        llm_issues: list[Issue] = []
        risk_factors: list[str] = []

        # Identify high-risk domains
        domain_issues = self._identify_risk_domains(requirement)
        rule_issues.extend(domain_issues)

        # Identify complexity indicators
        complexity_issues = self._identify_complexity(requirement)
        rule_issues.extend(complexity_issues)

        # Run pattern-based analysis
        if self._settings.enable_rule_based_analysis:
            pattern_issues = self._run_rule_based_analysis(requirement)
            rule_issues.extend(pattern_issues)
            logger.debug(f"Rule-based analysis found {len(rule_issues)} risk signals")

        # Run LLM analysis
        if self._settings.llm_available:
            try:
                response = self.llm_client.analyze_risk(requirement.full_text)
                llm_issues = self._parse_llm_issues(response, self.category)
                risk_factors = response.get("risk_factors", [])

                # Add recommended reviews as suggestions
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

                logger.debug(f"LLM analysis found {len(llm_issues)} risk issues")
            except LLMClientError as e:
                logger.warning(f"LLM analysis failed: {e}")

        # Merge and deduplicate
        all_issues = self._merge_issues(rule_issues, llm_issues)

        # Calculate risk score (inverted - higher score = lower risk)
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
                issues.append(
                    Issue(
                        severity=severity,
                        category=self.category,
                        location="requirement",
                        message=f"Touches {domain.replace('_', ' ')} domain - requires additional review",
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
        """
        Calculate risk score (0.0 = highest risk, 1.0 = lowest risk).

        This is inverted from typical risk scoring for consistency with other analyzers.
        """
        base_score = SCORE_PERFECT

        # Penalty based on issue severity
        weights = get_risk_severity_weights()
        for issue in issues:
            base_score -= weights.get(issue.severity.value, SEVERITY_WEIGHT_DEFAULT)

        # Additional penalty for multiple risk factors
        if len(risk_factors) > RISK_FACTORS_HIGH_THRESHOLD:
            base_score -= PENALTY_MULTIPLE_RISK_FACTORS

        return max(0.0, min(1.0, base_score))
