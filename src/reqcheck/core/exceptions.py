"""Custom exception hierarchy for reqcheck.

This module defines a hierarchy of exceptions used throughout the application
to provide clear, specific error handling with rich context.
"""

from typing import Any


class ReqcheckError(Exception):
    """Base exception for all reqcheck errors.

    All custom exceptions inherit from this class, making it easy to catch
    any reqcheck-specific error.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# Analysis Errors
# =============================================================================


class AnalysisError(ReqcheckError):
    """Base exception for analysis-related errors."""

    pass


class AnalysisTimeoutError(AnalysisError):
    """Raised when analysis exceeds the configured timeout."""

    def __init__(self, timeout_seconds: int, requirement_id: str | None = None):
        self.timeout_seconds = timeout_seconds
        self.requirement_id = requirement_id
        message = f"Analysis timed out after {timeout_seconds} seconds"
        details = {"timeout_seconds": timeout_seconds}
        if requirement_id:
            details["requirement_id"] = requirement_id
        super().__init__(message, details)


class AnalyzerError(AnalysisError):
    """Raised when a specific analyzer fails."""

    def __init__(
        self,
        analyzer_name: str,
        message: str,
        cause: Exception | None = None,
    ):
        self.analyzer_name = analyzer_name
        self.cause = cause
        details = {"analyzer": analyzer_name}
        if cause:
            details["cause"] = str(cause)
        super().__init__(message, details)


# =============================================================================
# LLM/API Errors
# =============================================================================


class LLMError(ReqcheckError):
    """Base exception for LLM-related errors."""

    pass


class LLMClientError(LLMError):
    """Raised when LLM API call fails."""

    def __init__(
        self,
        message: str,
        provider: str = "openai",
        status_code: int | None = None,
        retryable: bool = False,
    ):
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        details = {"provider": provider, "retryable": retryable}
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details)


class LLMConfigurationError(LLMError):
    """Raised when LLM is not properly configured."""

    def __init__(self, missing_config: str):
        self.missing_config = missing_config
        message = f"LLM configuration error: {missing_config} is not configured"
        super().__init__(message, {"missing_config": missing_config})


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or cannot be parsed."""

    def __init__(self, message: str, raw_response: str | None = None):
        self.raw_response = raw_response
        details = {}
        if raw_response:
            # Truncate for logging
            details["raw_response"] = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
        super().__init__(message, details)


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded."""

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        details = {}
        if retry_after:
            message += f", retry after {retry_after}s"
            details["retry_after"] = retry_after
        super().__init__(message, details)


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(ReqcheckError):
    """Base exception for validation errors."""

    pass


class RequirementValidationError(ValidationError):
    """Raised when a requirement fails validation."""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.value = value
        details = {"field": field}
        if value is not None:
            details["value"] = str(value)[:100]
        super().__init__(message, details)


class ConfigurationError(ValidationError):
    """Raised when configuration is invalid."""

    def __init__(self, setting: str, message: str, value: Any = None):
        self.setting = setting
        self.value = value
        details = {"setting": setting}
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, details)


# =============================================================================
# API Errors
# =============================================================================


class APIError(ReqcheckError):
    """Base exception for REST API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str | None = None,
    ):
        self.status_code = status_code
        self.error_code = error_code or "INTERNAL_ERROR"
        details = {"status_code": status_code, "error_code": self.error_code}
        super().__init__(message, details)


class RateLimitExceededError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, limit: str, retry_after: int | None = None):
        self.limit = limit
        self.retry_after = retry_after
        message = f"Rate limit exceeded: {limit}"
        super().__init__(message, status_code=429, error_code="RATE_LIMIT_EXCEEDED")
        self.details["limit"] = limit
        if retry_after:
            self.details["retry_after"] = retry_after


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401, error_code="AUTHENTICATION_REQUIRED")


class AuthorizationError(APIError):
    """Raised when API authorization fails."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=403, error_code="ACCESS_DENIED")


class RequestValidationError(APIError):
    """Raised when API request validation fails."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(message, status_code=422, error_code="VALIDATION_ERROR")
        self.details["field"] = field
