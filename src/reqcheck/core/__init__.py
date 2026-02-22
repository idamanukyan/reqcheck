"""Core components for the QA agent."""

from reqcheck.core.analyzer import RequirementsAnalyzer, analyze_requirement
from reqcheck.core.config import Settings, get_settings
from reqcheck.core.exceptions import (
    AnalysisError,
    AnalysisTimeoutError,
    AnalyzerError,
    APIError,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    LLMClientError,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    RateLimitExceededError,
    ReqcheckError,
    RequestValidationError,
    RequirementValidationError,
    ValidationError,
)
from reqcheck.core.models import AnalysisReport, Issue, Requirement

__all__ = [
    # Analyzer
    "RequirementsAnalyzer",
    "analyze_requirement",
    # Config
    "Settings",
    "get_settings",
    # Models
    "Requirement",
    "Issue",
    "AnalysisReport",
    # Exceptions - Base
    "ReqcheckError",
    # Exceptions - Analysis
    "AnalysisError",
    "AnalysisTimeoutError",
    "AnalyzerError",
    # Exceptions - LLM
    "LLMError",
    "LLMClientError",
    "LLMConfigurationError",
    "LLMResponseError",
    "LLMRateLimitError",
    # Exceptions - Validation
    "ValidationError",
    "RequirementValidationError",
    "ConfigurationError",
    # Exceptions - API
    "APIError",
    "RateLimitExceededError",
    "AuthenticationError",
    "AuthorizationError",
    "RequestValidationError",
]
