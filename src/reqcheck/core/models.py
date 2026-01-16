"""Data models for the QA agent."""

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from reqcheck.core.constants import get_overall_score_weights


class RequirementType(str, Enum):
    """Type of requirement document."""

    STORY = "story"
    BUG = "bug"
    TASK = "task"
    EPIC = "epic"
    FEATURE = "feature"


class Severity(str, Enum):
    """Issue severity levels."""

    BLOCKER = "blocker"  # Cannot proceed to development
    WARNING = "warning"  # Likely to cause rework
    SUGGESTION = "suggestion"  # Improvement opportunity


class IssueCategory(str, Enum):
    """Categories of quality issues."""

    AMBIGUITY = "ambiguity"
    COMPLETENESS = "completeness"
    TESTABILITY = "testability"
    RISK = "risk"


class Requirement(BaseModel):
    """Input model for a requirement/user story/ticket."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str = Field(..., min_length=1, description="Title of the requirement")
    description: str = Field(default="", description="Detailed description")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="List of acceptance criteria",
    )
    type: RequirementType = Field(
        default=RequirementType.STORY,
        description="Type of requirement",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (labels, priority, etc.)",
    )

    @property
    def full_text(self) -> str:
        """Combine all text fields for analysis."""
        parts = [self.title, self.description]
        if self.acceptance_criteria:
            parts.append("Acceptance Criteria:")
            parts.extend(f"- {ac}" for ac in self.acceptance_criteria)
        return "\n".join(parts)


class Issue(BaseModel):
    """A quality issue found during analysis."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    severity: Severity = Field(..., description="How critical is this issue")
    category: IssueCategory = Field(..., description="Type of quality issue")
    location: str = Field(..., description="Where the issue was found (field/line)")
    message: str = Field(..., description="Description of the problem")
    suggestion: str = Field(default="", description="How to fix the issue")
    evidence: str = Field(default="", description="The problematic text snippet")

    def __str__(self) -> str:
        icon = {"blocker": "[!]", "warning": "[?]", "suggestion": "[i]"}
        return f"{icon.get(self.severity.value, '[-]')} {self.message}"


class ScoreBreakdown(BaseModel):
    """Scores for different quality dimensions."""

    ambiguity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0=very ambiguous, 1=crystal clear"
    )
    completeness: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0=missing everything, 1=fully complete"
    )
    testability: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0=untestable, 1=fully testable"
    )
    overall: float = Field(default=0.0, ge=0.0, le=1.0, description="Weighted overall score")

    def calculate_overall(self) -> float:
        """Calculate weighted overall score."""
        weights = get_overall_score_weights()
        self.overall = (
            self.ambiguity * weights["ambiguity"]
            + self.completeness * weights["completeness"]
            + self.testability * weights["testability"]
        )
        return self.overall


class AnalysisReport(BaseModel):
    """Complete analysis report for a requirement."""

    requirement_id: str = Field(..., description="ID of the analyzed requirement")
    requirement_title: str = Field(default="", description="Title for reference")
    issues: list[Issue] = Field(default_factory=list, description="All issues found")
    scores: ScoreBreakdown = Field(
        default_factory=ScoreBreakdown,
        description="Quality scores",
    )
    summary: str = Field(default="", description="Executive summary")
    recommendations: list[str] = Field(
        default_factory=list,
        description="Prioritized recommendations",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional analysis metadata",
    )

    @property
    def blocker_count(self) -> int:
        """Count of blocker-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.BLOCKER)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def suggestion_count(self) -> int:
        """Count of suggestion-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.SUGGESTION)

    @property
    def is_ready_for_dev(self) -> bool:
        """Check if requirement is ready for development (no blockers)."""
        return self.blocker_count == 0

    def issues_by_category(self, category: IssueCategory) -> list[Issue]:
        """Filter issues by category."""
        return [i for i in self.issues if i.category == category]


class PatternMatch(BaseModel):
    """A pattern match from rule-based analysis."""

    pattern_name: str = Field(..., description="Name of the matched pattern")
    matched_text: str = Field(..., description="The text that matched")
    start_pos: int = Field(..., description="Start position in source text")
    end_pos: int = Field(..., description="End position in source text")
    severity: Severity = Field(..., description="Suggested severity")
    category: IssueCategory = Field(..., description="Issue category")
    message_template: str = Field(..., description="Message template for this pattern")

    def to_issue(self, location: str) -> Issue:
        """Convert pattern match to an Issue."""
        return Issue(
            severity=self.severity,
            category=self.category,
            location=location,
            message=self.message_template.format(text=self.matched_text),
            evidence=self.matched_text,
        )
