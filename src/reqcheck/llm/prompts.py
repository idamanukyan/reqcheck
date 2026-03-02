"""Prompt templates for LLM-powered analysis.

This module contains all prompt templates with versioning for change tracking.
Each prompt has a version number that should be incremented when the prompt
content changes significantly (affecting LLM behavior).

Version History:
- v1.0.0 (2024-01): Initial release
- v1.1.0 (2024-02): Added prompt versioning support
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PromptType(Enum):
    """Types of analysis prompts."""

    SYSTEM = "system"
    AMBIGUITY = "ambiguity"
    COMPLETENESS = "completeness"
    TESTABILITY = "testability"
    RISK = "risk"
    SUMMARY = "summary"


@dataclass(frozen=True)
class PromptVersion:
    """Version information for a prompt template."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "version_string": str(self),
        }


# Current versions for each prompt type
PROMPT_VERSIONS: dict[PromptType, PromptVersion] = {
    PromptType.SYSTEM: PromptVersion(1, 1, 0),
    PromptType.AMBIGUITY: PromptVersion(1, 1, 0),
    PromptType.COMPLETENESS: PromptVersion(1, 1, 0),
    PromptType.TESTABILITY: PromptVersion(1, 1, 0),
    PromptType.RISK: PromptVersion(1, 1, 0),
    PromptType.SUMMARY: PromptVersion(1, 1, 0),
}


SYSTEM_PROMPT = """You are an expert QA architect and requirements analyst. Your task is to analyze software requirements, user stories, and tickets for quality issues that could lead to development problems, bugs, or rework.

You have deep expertise in:
- Software requirements engineering
- Agile methodologies and user story best practices
- Test-driven development and acceptance criteria
- Risk assessment and quality assurance

Be specific and actionable in your analysis. Focus on issues that would actually cause problems during development or testing. Avoid generic advice."""

AMBIGUITY_ANALYSIS_PROMPT = """Analyze the following requirement for ambiguity issues.

Look for:
1. Referential ambiguity: Pronouns or references without clear antecedents ("it", "this", "the system")
2. Scope ambiguity: Quantifiers without explicit bounds ("all users", "some data")
3. Temporal ambiguity: Timing without specific constraints ("immediately", "in real-time")
4. Lexical ambiguity: Terms that could have multiple meanings in this context
5. Conditional ambiguity: If/when statements without complete condition handling

REQUIREMENT:
{requirement_text}

Respond with a JSON object containing:
{{
  "issues": [
    {{
      "severity": "blocker|warning|suggestion",
      "location": "title|description|acceptance_criteria[N]",
      "message": "specific description of the ambiguity",
      "suggestion": "how to fix it",
      "evidence": "the ambiguous text snippet"
    }}
  ],
  "ambiguity_score": 0.0-1.0 (1.0 = no ambiguity, 0.0 = highly ambiguous)
}}

Only include real issues. If the requirement is clear, return empty issues array with high score."""

COMPLETENESS_ANALYSIS_PROMPT = """Analyze the following requirement for completeness issues.

Check for:
1. Missing happy path: Is the main success scenario fully described?
2. Missing error handling: What happens when things go wrong?
3. Missing edge cases: Boundary conditions, empty states, limits
4. Missing integration context: How does this interact with other features?
5. Missing user context: Which user roles? What permissions?
6. Missing data specifications: Input validation, data types, formats

REQUIREMENT:
{requirement_text}

Respond with a JSON object containing:
{{
  "issues": [
    {{
      "severity": "blocker|warning|suggestion",
      "location": "title|description|acceptance_criteria|missing",
      "message": "what is missing and why it matters",
      "suggestion": "what should be added",
      "evidence": "relevant context from the requirement"
    }}
  ],
  "completeness_score": 0.0-1.0 (1.0 = complete, 0.0 = missing everything),
  "missing_sections": ["list of missing elements"]
}}

Focus on gaps that would actually block development or cause defects."""

TESTABILITY_ANALYSIS_PROMPT = """Analyze the following requirement for testability issues.

Evaluate:
1. Can each acceptance criterion be converted to a pass/fail test?
2. Are success and failure states explicitly defined?
3. Are there measurable outcomes (not just "works correctly")?
4. Are boundary conditions testable?
5. Are there implicit behaviors that aren't testable from the text?

REQUIREMENT:
{requirement_text}

Respond with a JSON object containing:
{{
  "issues": [
    {{
      "severity": "blocker|warning|suggestion",
      "location": "acceptance_criteria[N]|description",
      "message": "why this is not testable",
      "suggestion": "how to make it testable",
      "evidence": "the untestable text"
    }}
  ],
  "testability_score": 0.0-1.0 (1.0 = fully testable, 0.0 = untestable),
  "suggested_test_scenarios": ["list of test cases this should cover but doesn't explicitly mention"]
}}

Consider: Could a QA engineer who wasn't in the planning meeting write tests from this alone?"""

RISK_ANALYSIS_PROMPT = """Analyze the following requirement for delivery and quality risks.

Assess:
1. Security risks: Authentication, authorization, data handling
2. Integration risks: External systems, APIs, third-party dependencies
3. Complexity risks: Multiple conditional paths, state management
4. Data risks: Migrations, transformations, compliance (GDPR, PCI)
5. Performance risks: Scale, latency, resource usage
6. Dependency risks: Cross-team, external vendor, technical debt

REQUIREMENT:
{requirement_text}

Respond with a JSON object containing:
{{
  "issues": [
    {{
      "severity": "blocker|warning|suggestion",
      "location": "title|description|acceptance_criteria[N]",
      "message": "description of the risk",
      "suggestion": "mitigation approach",
      "evidence": "text indicating the risk"
    }}
  ],
  "risk_level": "low|medium|high|critical",
  "risk_factors": ["list of identified risk factors"],
  "recommended_reviews": ["security review", "architecture review", etc.]
}}

Focus on risks that can be identified from the requirement text alone."""

SUMMARY_PROMPT = """Based on the following analysis results, provide a concise executive summary.

REQUIREMENT TITLE: {title}

ISSUES FOUND:
{issues_summary}

SCORES:
- Ambiguity: {ambiguity_score}/1.0
- Completeness: {completeness_score}/1.0
- Testability: {testability_score}/1.0

Provide a JSON response:
{{
  "summary": "2-3 sentence executive summary of the requirement quality",
  "recommendations": ["prioritized list of 3-5 most important actions"],
  "ready_for_development": true/false,
  "confidence": "high|medium|low"
}}

Be direct and actionable. If the requirement is ready for development, say so."""


class PromptTemplates:
    """Container for all prompt templates with versioning support.

    Each prompt has an associated version that should be incremented when
    the prompt content changes in ways that could affect LLM behavior.

    Usage:
        # Get prompt content
        prompt = PromptTemplates.format_ambiguity(requirement_text)

        # Get version info
        version = PromptTemplates.get_version(PromptType.AMBIGUITY)

        # Get all versions for logging/tracking
        versions = PromptTemplates.get_all_versions()
    """

    SYSTEM = SYSTEM_PROMPT
    AMBIGUITY = AMBIGUITY_ANALYSIS_PROMPT
    COMPLETENESS = COMPLETENESS_ANALYSIS_PROMPT
    TESTABILITY = TESTABILITY_ANALYSIS_PROMPT
    RISK = RISK_ANALYSIS_PROMPT
    SUMMARY = SUMMARY_PROMPT

    # Map prompt types to their templates
    _TEMPLATES: dict[PromptType, str] = {
        PromptType.SYSTEM: SYSTEM_PROMPT,
        PromptType.AMBIGUITY: AMBIGUITY_ANALYSIS_PROMPT,
        PromptType.COMPLETENESS: COMPLETENESS_ANALYSIS_PROMPT,
        PromptType.TESTABILITY: TESTABILITY_ANALYSIS_PROMPT,
        PromptType.RISK: RISK_ANALYSIS_PROMPT,
        PromptType.SUMMARY: SUMMARY_PROMPT,
    }

    @classmethod
    def get_version(cls, prompt_type: PromptType) -> PromptVersion:
        """Get the version of a specific prompt type."""
        return PROMPT_VERSIONS[prompt_type]

    @classmethod
    def get_all_versions(cls) -> dict[str, str]:
        """Get all prompt versions as a dictionary.

        Returns:
            Dictionary mapping prompt type names to version strings.
        """
        return {pt.value: str(PROMPT_VERSIONS[pt]) for pt in PromptType}

    @classmethod
    def get_version_info(cls) -> dict[str, Any]:
        """Get detailed version information for all prompts.

        Returns:
            Dictionary with prompt type names mapping to version details.
        """
        return {pt.value: PROMPT_VERSIONS[pt].to_dict() for pt in PromptType}

    @classmethod
    def get_prompt_hash(cls, prompt_type: PromptType) -> str:
        """Get a hash of the prompt content for cache keying.

        This can be used to invalidate caches when prompt content changes.
        """
        import hashlib

        content = cls._TEMPLATES[prompt_type]
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @classmethod
    def format_ambiguity(cls, requirement_text: str) -> str:
        return cls.AMBIGUITY.format(requirement_text=requirement_text)

    @classmethod
    def format_completeness(cls, requirement_text: str) -> str:
        return cls.COMPLETENESS.format(requirement_text=requirement_text)

    @classmethod
    def format_testability(cls, requirement_text: str) -> str:
        return cls.TESTABILITY.format(requirement_text=requirement_text)

    @classmethod
    def format_risk(cls, requirement_text: str) -> str:
        return cls.RISK.format(requirement_text=requirement_text)

    @classmethod
    def format_summary(
        cls,
        title: str,
        issues_summary: str,
        ambiguity_score: float,
        completeness_score: float,
        testability_score: float,
    ) -> str:
        return cls.SUMMARY.format(
            title=title,
            issues_summary=issues_summary,
            ambiguity_score=f"{ambiguity_score:.2f}",
            completeness_score=f"{completeness_score:.2f}",
            testability_score=f"{testability_score:.2f}",
        )


def get_prompt_versions() -> dict[str, str]:
    """Convenience function to get all prompt versions.

    Returns:
        Dictionary mapping prompt type names to version strings.
    """
    return PromptTemplates.get_all_versions()
