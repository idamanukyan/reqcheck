"""Core components for the QA agent."""

from reqcheck.core.analyzer import RequirementsAnalyzer, analyze_requirement
from reqcheck.core.config import Settings, get_settings
from reqcheck.core.models import AnalysisReport, Issue, Requirement

__all__ = [
    "RequirementsAnalyzer",
    "analyze_requirement",
    "Settings",
    "get_settings",
    "Requirement",
    "Issue",
    "AnalysisReport",
]
