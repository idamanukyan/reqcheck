"""Configuration management for reqcheck."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="REQCHECK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Selection
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="LLM provider to use (openai or anthropic)",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use",
    )
    openai_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Model temperature (lower = more deterministic)",
    )
    openai_max_tokens: int = Field(
        default=2000,
        ge=100,
        le=16000,
        description="Maximum tokens in response",
    )

    # Anthropic Configuration
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Anthropic model to use",
    )
    anthropic_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Model temperature for Anthropic",
    )
    anthropic_max_tokens: int = Field(
        default=2000,
        ge=100,
        le=16000,
        description="Maximum tokens for Anthropic response",
    )

    # LLM Retry Configuration
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts for LLM API calls",
    )
    llm_retry_base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Base delay in seconds for exponential backoff",
    )
    llm_retry_max_delay: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Maximum delay in seconds between retries",
    )

    # LLM Cache Configuration
    llm_cache_enabled: bool = Field(
        default=True,
        description="Enable caching of LLM responses",
    )
    llm_cache_ttl_seconds: float = Field(
        default=3600.0,
        ge=60.0,
        le=86400.0,
        description="Time-to-live for cached LLM responses in seconds",
    )
    llm_cache_max_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum number of cached LLM responses",
    )

    # Token Usage Tracking
    track_token_usage: bool = Field(
        default=True,
        description="Track token usage and estimate costs",
    )

    # Custom Patterns Configuration
    custom_patterns_file: str = Field(
        default="",
        description="Path to custom patterns YAML file",
    )
    custom_weasel_words: str = Field(
        default="",
        description="Comma-separated list of additional weasel words",
    )
    custom_forbidden_terms: str = Field(
        default="",
        description="Comma-separated list of forbidden terms (blockers)",
    )

    # Analysis Configuration
    enable_llm_analysis: bool = Field(
        default=True,
        description="Enable LLM-powered analysis (requires API key)",
    )
    enable_rule_based_analysis: bool = Field(
        default=True,
        description="Enable rule-based pattern matching",
    )
    analysis_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Analysis timeout in seconds",
    )

    # Severity Thresholds
    min_severity: Literal["blocker", "warning", "suggestion"] = Field(
        default="suggestion",
        description="Minimum severity to report",
    )

    # Output Configuration
    default_output_format: Literal["json", "markdown", "summary"] = Field(
        default="markdown",
        description="Default output format",
    )
    include_evidence: bool = Field(
        default=True,
        description="Include evidence snippets in output",
    )
    include_suggestions: bool = Field(
        default=True,
        description="Include fix suggestions in output",
    )

    # API Server Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API server port")

    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable API rate limiting",
    )
    rate_limit_default: str = Field(
        default="60/minute",
        description="Default rate limit (e.g., '60/minute', '100/hour')",
    )
    rate_limit_analyze: str = Field(
        default="30/minute",
        description="Rate limit for /analyze endpoints",
    )
    rate_limit_batch: str = Field(
        default="10/minute",
        description="Rate limit for /analyze/batch endpoint",
    )

    # CORS Configuration
    cors_allowed_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins or '*' for all",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
    )
    cors_allow_methods: str = Field(
        default="*",
        description="Comma-separated list of allowed HTTP methods or '*' for all",
    )
    cors_allow_headers: str = Field(
        default="*",
        description="Comma-separated list of allowed headers or '*' for all",
    )

    # Authentication Configuration
    auth_enabled: bool = Field(
        default=False,
        description="Enable API key authentication (disabled by default)",
    )
    api_keys: str = Field(
        default="",
        description="Comma-separated list of valid API keys",
    )
    auth_header_name: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication",
    )

    # Request Size Limits
    max_title_length: int = Field(
        default=500,
        ge=10,
        le=2000,
        description="Maximum character length for requirement title",
    )
    max_description_length: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum character length for requirement description",
    )
    max_acceptance_criteria_count: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of acceptance criteria per requirement",
    )
    max_acceptance_criteria_length: int = Field(
        default=2000,
        ge=50,
        le=10000,
        description="Maximum character length for each acceptance criterion",
    )
    max_metadata_size_bytes: int = Field(
        default=10240,
        ge=256,
        le=102400,
        description="Maximum size of metadata JSON in bytes",
    )

    @property
    def llm_available(self) -> bool:
        """Check if LLM analysis is available."""
        if not self.enable_llm_analysis:
            return False
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        elif self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return False

    @property
    def current_api_key(self) -> str:
        """Get the API key for the current provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return ""

    @property
    def custom_weasel_words_list(self) -> list[str]:
        """Parse custom weasel words into a list."""
        if not self.custom_weasel_words:
            return []
        return [w.strip() for w in self.custom_weasel_words.split(",") if w.strip()]

    @property
    def custom_forbidden_terms_list(self) -> list[str]:
        """Parse custom forbidden terms into a list."""
        if not self.custom_forbidden_terms:
            return []
        return [t.strip() for t in self.custom_forbidden_terms.split(",") if t.strip()]

    def load_custom_patterns(self) -> dict[str, Any]:
        """Load custom patterns from YAML file if configured."""
        if not self.custom_patterns_file:
            return {}

        path = Path(self.custom_patterns_file)
        if not path.exists():
            return {}

        try:
            import yaml

            with open(path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # YAML not installed
            return {}
        except Exception:
            return {}

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        if self.cors_allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def cors_methods_list(self) -> list[str]:
        """Parse CORS methods into a list."""
        if self.cors_allow_methods == "*":
            return ["*"]
        return [m.strip() for m in self.cors_allow_methods.split(",") if m.strip()]

    @property
    def cors_headers_list(self) -> list[str]:
        """Parse CORS headers into a list."""
        if self.cors_allow_headers == "*":
            return ["*"]
        return [h.strip() for h in self.cors_allow_headers.split(",") if h.strip()]

    @property
    def api_keys_set(self) -> set[str]:
        """Parse API keys into a set for O(1) lookup."""
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reset_settings() -> None:
    """Clear the settings cache (useful for testing)."""
    get_settings.cache_clear()
