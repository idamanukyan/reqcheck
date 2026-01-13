"""Configuration management for reqcheck."""

from functools import lru_cache
from typing import Literal

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

    @property
    def llm_available(self) -> bool:
        """Check if LLM analysis is available."""
        return bool(self.openai_api_key) and self.enable_llm_analysis


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reset_settings() -> None:
    """Clear the settings cache (useful for testing)."""
    get_settings.cache_clear()
