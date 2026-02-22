"""Tests for the LLM client wrapper."""

import json
from unittest.mock import MagicMock, patch

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from reqcheck.core.config import Settings
from reqcheck.core.exceptions import (
    LLMClientError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
)
from reqcheck.llm.client import LLMClient


@pytest.fixture
def settings_with_api_key():
    """Create test settings with API key configured."""
    return Settings(
        openai_api_key="test-api-key-12345",
        openai_model="gpt-4o-mini",
        openai_temperature=0.3,
        openai_max_tokens=2000,
    )


@pytest.fixture
def settings_without_api_key():
    """Create test settings without API key."""
    return Settings(
        openai_api_key="",
    )


@pytest.fixture
def llm_client(settings_with_api_key):
    """Create LLM client with test settings."""
    return LLMClient(settings_with_api_key)


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "issues": [
            {
                "severity": "warning",
                "location": "description",
                "message": "Test issue found",
                "suggestion": "Fix it",
                "evidence": "test text",
            }
        ],
        "ambiguity_score": 0.7,
    })
    return mock_response


class TestLLMClientInitialization:
    """Tests for LLM client initialization."""

    def test_init_with_settings(self, settings_with_api_key):
        """Test client initialization with provided settings."""
        client = LLMClient(settings_with_api_key)
        assert client._settings == settings_with_api_key
        assert client._client is None  # Lazy initialization

    def test_init_without_settings(self):
        """Test client initialization uses default settings."""
        with patch("reqcheck.llm.client.get_settings") as mock_get_settings:
            mock_settings = Settings(openai_api_key="default-key")
            mock_get_settings.return_value = mock_settings
            client = LLMClient()
            mock_get_settings.assert_called_once()
            assert client._settings == mock_settings

    def test_lazy_client_initialization(self, settings_with_api_key):
        """Test that OpenAI client is not created until accessed."""
        client = LLMClient(settings_with_api_key)
        assert client._client is None

        with patch("reqcheck.llm.client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            _ = client.client
            mock_openai.assert_called_once_with(api_key="test-api-key-12345")

    def test_client_property_caches_instance(self, settings_with_api_key):
        """Test that client property returns cached instance."""
        client = LLMClient(settings_with_api_key)

        with patch("reqcheck.llm.client.OpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            first_call = client.client
            second_call = client.client

            assert first_call is second_call
            mock_openai.assert_called_once()


class TestAPIKeyValidation:
    """Tests for API key validation."""

    def test_missing_api_key_raises_error(self, settings_without_api_key):
        """Test that missing API key raises LLMConfigurationError."""
        client = LLMClient(settings_without_api_key)

        with pytest.raises(LLMConfigurationError) as exc_info:
            _ = client.client

        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "not configured" in str(exc_info.value)

    def test_empty_api_key_raises_error(self):
        """Test that empty string API key raises LLMConfigurationError."""
        settings = Settings(openai_api_key="")
        client = LLMClient(settings)

        with pytest.raises(LLMConfigurationError) as exc_info:
            _ = client.client

        assert "not configured" in str(exc_info.value)

    def test_whitespace_api_key_is_treated_as_configured(self):
        """Test that whitespace-only API key is treated as configured (OpenAI validates)."""
        settings = Settings(openai_api_key="   ")
        client = LLMClient(settings)

        # Whitespace is truthy, so client considers it configured
        # OpenAI's API will reject invalid keys at call time
        assert client.is_available() is True


class TestAPICallErrorHandling:
    """Tests for API call error handling."""

    def test_api_timeout_error(self, llm_client):
        """Test handling of API timeout errors."""
        with patch.object(llm_client, "_client", create=True) as mock_client:
            llm_client._client = mock_client
            mock_client.chat.completions.create.side_effect = APITimeoutError(
                request=MagicMock()
            )

            with pytest.raises(LLMClientError) as exc_info:
                llm_client._call_api("test prompt")

            assert "API call failed" in str(exc_info.value)

    def test_api_connection_error(self, llm_client):
        """Test handling of API connection errors."""
        with patch.object(llm_client, "_client", create=True) as mock_client:
            llm_client._client = mock_client
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )

            with pytest.raises(LLMClientError) as exc_info:
                llm_client._call_api("test prompt")

            assert "API call failed" in str(exc_info.value)

    def test_rate_limit_error(self, llm_client):
        """Test handling of rate limit errors."""
        with patch.object(llm_client, "_client", create=True) as mock_client:
            llm_client._client = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_client.chat.completions.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "Rate limit exceeded"}},
            )

            with pytest.raises(LLMRateLimitError) as exc_info:
                llm_client._call_api("test prompt")

            assert "Rate limit exceeded" in str(exc_info.value)

    def test_empty_response_error(self, llm_client):
        """Test handling of empty API response."""
        with patch.object(llm_client, "_client", create=True) as mock_client:
            llm_client._client = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_client.chat.completions.create.return_value = mock_response

            with pytest.raises(LLMResponseError) as exc_info:
                llm_client._call_api("test prompt")

            assert "Empty response" in str(exc_info.value)

    def test_successful_api_call(self, llm_client, mock_openai_response):
        """Test successful API call returns content."""
        with patch.object(llm_client, "_client", create=True) as mock_client:
            llm_client._client = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            result = llm_client._call_api("test prompt")

            assert "issues" in result
            assert "ambiguity_score" in result


class TestJSONParsing:
    """Tests for JSON response parsing."""

    def test_valid_json_parsing(self, llm_client):
        """Test parsing of valid JSON response."""
        valid_json = json.dumps({
            "issues": [],
            "ambiguity_score": 0.9,
        })

        result = llm_client._parse_json_response(valid_json)

        assert result["issues"] == []
        assert result["ambiguity_score"] == 0.9

    def test_invalid_json_raises_error(self, llm_client):
        """Test that invalid JSON raises LLMResponseError."""
        invalid_json = "{ this is not valid json }"

        with pytest.raises(LLMResponseError) as exc_info:
            llm_client._parse_json_response(invalid_json)

        assert "Invalid JSON response" in str(exc_info.value)

    def test_malformed_json_raises_error(self, llm_client):
        """Test that malformed JSON raises LLMResponseError."""
        malformed_json = '{"issues": [, "score": 0.5}'

        with pytest.raises(LLMResponseError) as exc_info:
            llm_client._parse_json_response(malformed_json)

        assert "Invalid JSON response" in str(exc_info.value)

    def test_empty_string_raises_error(self, llm_client):
        """Test that empty string raises LLMResponseError."""
        with pytest.raises(LLMResponseError) as exc_info:
            llm_client._parse_json_response("")

        assert "Invalid JSON response" in str(exc_info.value)

    def test_json_with_unicode(self, llm_client):
        """Test parsing JSON with unicode characters."""
        unicode_json = json.dumps({
            "issues": [{"message": "Текст на русском 日本語"}],
            "score": 0.8,
        })

        result = llm_client._parse_json_response(unicode_json)

        assert "Текст на русском" in result["issues"][0]["message"]


class TestAnalyzeMethods:
    """Tests for analyze methods."""

    def test_analyze_ambiguity(self, llm_client):
        """Test analyze_ambiguity method."""
        mock_response = json.dumps({
            "issues": [
                {
                    "severity": "warning",
                    "location": "description",
                    "message": "Vague term detected",
                    "suggestion": "Be more specific",
                    "evidence": "appropriately",
                }
            ],
            "ambiguity_score": 0.6,
        })

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_ambiguity("Test requirement text")

            assert len(result["issues"]) == 1
            assert result["ambiguity_score"] == 0.6

    def test_analyze_completeness(self, llm_client):
        """Test analyze_completeness method."""
        mock_response = json.dumps({
            "issues": [
                {
                    "severity": "blocker",
                    "location": "missing",
                    "message": "Missing error handling",
                    "suggestion": "Add error scenarios",
                    "evidence": "N/A",
                }
            ],
            "completeness_score": 0.5,
            "missing_sections": ["error handling", "edge cases"],
        })

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_completeness("Test requirement text")

            assert len(result["issues"]) == 1
            assert result["completeness_score"] == 0.5
            assert "error handling" in result["missing_sections"]

    def test_analyze_testability(self, llm_client):
        """Test analyze_testability method."""
        mock_response = json.dumps({
            "issues": [],
            "testability_score": 0.9,
            "suggested_test_scenarios": ["happy path", "error case"],
        })

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_testability("Test requirement text")

            assert result["issues"] == []
            assert result["testability_score"] == 0.9
            assert len(result["suggested_test_scenarios"]) == 2

    def test_analyze_risk(self, llm_client):
        """Test analyze_risk method."""
        mock_response = json.dumps({
            "issues": [
                {
                    "severity": "warning",
                    "location": "description",
                    "message": "Security risk identified",
                    "suggestion": "Implement auth check",
                    "evidence": "payment processing",
                }
            ],
            "risk_level": "high",
            "risk_factors": ["payment handling", "PCI compliance"],
            "recommended_reviews": ["security review"],
        })

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_risk("Process payment for user")

            assert result["risk_level"] == "high"
            assert "payment handling" in result["risk_factors"]

    def test_generate_summary(self, llm_client):
        """Test generate_summary method."""
        mock_response = json.dumps({
            "summary": "The requirement has moderate issues.",
            "recommendations": ["Clarify terms", "Add acceptance criteria"],
            "ready_for_development": False,
            "confidence": "high",
        })

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.generate_summary(
                title="User Login",
                issues_summary="2 warnings, 1 blocker",
                ambiguity_score=0.7,
                completeness_score=0.6,
                testability_score=0.8,
            )

            assert result["ready_for_development"] is False
            assert len(result["recommendations"]) == 2


class TestIsAvailable:
    """Tests for is_available method."""

    def test_is_available_with_api_key(self, settings_with_api_key):
        """Test is_available returns True when API key is set."""
        client = LLMClient(settings_with_api_key)
        assert client.is_available() is True

    def test_is_available_without_api_key(self, settings_without_api_key):
        """Test is_available returns False when API key is missing."""
        client = LLMClient(settings_without_api_key)
        assert client.is_available() is False

    def test_is_available_with_empty_api_key(self):
        """Test is_available returns False with empty API key."""
        settings = Settings(openai_api_key="")
        client = LLMClient(settings)
        assert client.is_available() is False


class TestPromptFormatting:
    """Tests for prompt formatting integration."""

    def test_ambiguity_prompt_includes_requirement(self, llm_client):
        """Test that ambiguity prompt includes the requirement text."""
        requirement_text = "Users should login appropriately"

        with patch.object(llm_client, "_call_api") as mock_call:
            mock_call.return_value = json.dumps({"issues": [], "ambiguity_score": 1.0})
            llm_client.analyze_ambiguity(requirement_text)

            called_prompt = mock_call.call_args[0][0]
            assert requirement_text in called_prompt

    def test_completeness_prompt_includes_requirement(self, llm_client):
        """Test that completeness prompt includes the requirement text."""
        requirement_text = "Feature handles all data"

        with patch.object(llm_client, "_call_api") as mock_call:
            mock_call.return_value = json.dumps({
                "issues": [],
                "completeness_score": 1.0,
                "missing_sections": [],
            })
            llm_client.analyze_completeness(requirement_text)

            called_prompt = mock_call.call_args[0][0]
            assert requirement_text in called_prompt

    def test_summary_prompt_includes_all_parameters(self, llm_client):
        """Test that summary prompt includes all provided parameters."""
        with patch.object(llm_client, "_call_api") as mock_call:
            mock_call.return_value = json.dumps({
                "summary": "Test",
                "recommendations": [],
                "ready_for_development": True,
                "confidence": "high",
            })
            llm_client.generate_summary(
                title="Test Title",
                issues_summary="No issues",
                ambiguity_score=0.9,
                completeness_score=0.8,
                testability_score=0.7,
            )

            called_prompt = mock_call.call_args[0][0]
            assert "Test Title" in called_prompt
            assert "No issues" in called_prompt
            assert "0.90" in called_prompt  # Formatted score


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_requirement_text(self, llm_client):
        """Test handling of very long requirement text."""
        long_text = "A" * 10000
        mock_response = json.dumps({"issues": [], "ambiguity_score": 0.9})

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_ambiguity(long_text)
            assert result["ambiguity_score"] == 0.9

    def test_special_characters_in_requirement(self, llm_client):
        """Test handling of special characters in requirement text."""
        special_text = 'Requirement with "quotes" and {braces} and <tags>'
        mock_response = json.dumps({"issues": [], "ambiguity_score": 0.9})

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_ambiguity(special_text)
            assert result["ambiguity_score"] == 0.9

    def test_newlines_in_requirement(self, llm_client):
        """Test handling of newlines in requirement text."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        mock_response = json.dumps({"issues": [], "ambiguity_score": 0.9})

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_ambiguity(multiline_text)
            assert result["ambiguity_score"] == 0.9

    def test_empty_issues_array(self, llm_client):
        """Test handling of response with empty issues array."""
        mock_response = json.dumps({
            "issues": [],
            "ambiguity_score": 1.0,
        })

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_ambiguity("Perfect requirement")
            assert result["issues"] == []
            assert result["ambiguity_score"] == 1.0

    def test_response_with_extra_fields(self, llm_client):
        """Test handling of response with unexpected extra fields."""
        mock_response = json.dumps({
            "issues": [],
            "ambiguity_score": 0.9,
            "unexpected_field": "should be preserved",
            "another_field": 123,
        })

        with patch.object(llm_client, "_call_api", return_value=mock_response):
            result = llm_client.analyze_ambiguity("Test")
            assert result["unexpected_field"] == "should be preserved"
            assert result["another_field"] == 123


class TestRetryLogic:
    """Tests for LLM client retry logic with exponential backoff."""

    @pytest.fixture
    def settings_with_retries(self):
        """Create test settings with retry configuration."""
        return Settings(
            openai_api_key="test-api-key-12345",
            llm_max_retries=3,
            llm_retry_base_delay=0.1,  # Minimum allowed value
            llm_retry_max_delay=1.0,   # Minimum allowed value
        )

    @pytest.fixture
    def settings_no_retries(self):
        """Create test settings with no retries."""
        return Settings(
            openai_api_key="test-api-key-12345",
            llm_max_retries=0,
        )

    @pytest.fixture
    def client_with_retries(self, settings_with_retries):
        """Create LLM client with retry settings."""
        return LLMClient(settings_with_retries)

    @pytest.fixture
    def client_no_retries(self, settings_no_retries):
        """Create LLM client with no retries."""
        return LLMClient(settings_no_retries)

    def test_successful_on_first_attempt(self, client_with_retries, mock_openai_response):
        """Test that successful first attempt does not trigger retries."""
        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            result = client_with_retries._call_api("test prompt")

            assert "issues" in result
            mock_client.chat.completions.create.assert_called_once()

    def test_retry_on_rate_limit_error(self, client_with_retries, mock_openai_response):
        """Test retry on RateLimitError."""
        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 429

            # Fail twice, succeed on third attempt
            mock_client.chat.completions.create.side_effect = [
                RateLimitError(
                    message="Rate limit exceeded",
                    response=mock_response,
                    body={"error": {"message": "Rate limit exceeded"}},
                ),
                RateLimitError(
                    message="Rate limit exceeded",
                    response=mock_response,
                    body={"error": {"message": "Rate limit exceeded"}},
                ),
                mock_openai_response,
            ]

            with patch("time.sleep"):  # Don't actually sleep in tests
                result = client_with_retries._call_api("test prompt")

            assert "issues" in result
            assert mock_client.chat.completions.create.call_count == 3

    def test_retry_on_timeout_error(self, client_with_retries, mock_openai_response):
        """Test retry on APITimeoutError."""
        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client

            # Fail once, succeed on second attempt
            mock_client.chat.completions.create.side_effect = [
                APITimeoutError(request=MagicMock()),
                mock_openai_response,
            ]

            with patch("time.sleep"):
                result = client_with_retries._call_api("test prompt")

            assert "issues" in result
            assert mock_client.chat.completions.create.call_count == 2

    def test_retry_on_connection_error(self, client_with_retries, mock_openai_response):
        """Test retry on APIConnectionError."""
        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client

            mock_client.chat.completions.create.side_effect = [
                APIConnectionError(request=MagicMock()),
                mock_openai_response,
            ]

            with patch("time.sleep"):
                result = client_with_retries._call_api("test prompt")

            assert "issues" in result
            assert mock_client.chat.completions.create.call_count == 2

    def test_retry_on_internal_server_error(self, client_with_retries, mock_openai_response):
        """Test retry on InternalServerError (500)."""
        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 500

            mock_client.chat.completions.create.side_effect = [
                InternalServerError(
                    message="Internal server error",
                    response=mock_response,
                    body={"error": {"message": "Internal server error"}},
                ),
                mock_openai_response,
            ]

            with patch("time.sleep"):
                result = client_with_retries._call_api("test prompt")

            assert "issues" in result
            assert mock_client.chat.completions.create.call_count == 2

    def test_no_retry_on_authentication_error(self, client_with_retries):
        """Test no retry on AuthenticationError (non-retryable)."""
        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 401

            mock_client.chat.completions.create.side_effect = AuthenticationError(
                message="Invalid API key",
                response=mock_response,
                body={"error": {"message": "Invalid API key"}},
            )

            with pytest.raises(LLMClientError) as exc_info:
                client_with_retries._call_api("test prompt")

            assert "API call failed" in str(exc_info.value)
            # Should only attempt once (no retries)
            mock_client.chat.completions.create.assert_called_once()

    def test_no_retry_on_bad_request_error(self, client_with_retries):
        """Test no retry on BadRequestError (non-retryable)."""
        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 400

            mock_client.chat.completions.create.side_effect = BadRequestError(
                message="Bad request",
                response=mock_response,
                body={"error": {"message": "Bad request"}},
            )

            with pytest.raises(LLMClientError) as exc_info:
                client_with_retries._call_api("test prompt")

            assert "API call failed" in str(exc_info.value)
            mock_client.chat.completions.create.assert_called_once()

    def test_all_retries_exhausted(self, client_with_retries):
        """Test error raised after all retries exhausted."""
        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 429

            # Always fail with rate limit error
            mock_client.chat.completions.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "Rate limit exceeded"}},
            )

            with patch("time.sleep"):
                with pytest.raises(LLMRateLimitError):
                    client_with_retries._call_api("test prompt")

            # 1 initial + 3 retries = 4 total attempts
            assert mock_client.chat.completions.create.call_count == 4

    def test_no_retries_when_disabled(self, client_no_retries):
        """Test no retries when max_retries is 0."""
        with patch.object(client_no_retries, "_client", create=True) as mock_client:
            client_no_retries._client = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 429

            mock_client.chat.completions.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "Rate limit exceeded"}},
            )

            with pytest.raises(LLMRateLimitError):
                client_no_retries._call_api("test prompt")

            mock_client.chat.completions.create.assert_called_once()

    def test_backoff_delay_calculation(self, client_with_retries):
        """Test exponential backoff delay calculation."""
        # Attempt 0: base_delay * 2^0 = 0.1 * 1 = 0.1s (±jitter)
        delay_0 = client_with_retries._calculate_backoff_delay(0)
        assert 0.075 <= delay_0 <= 0.125  # 0.1 ± 25%

        # Attempt 1: base_delay * 2^1 = 0.1 * 2 = 0.2s (±jitter)
        delay_1 = client_with_retries._calculate_backoff_delay(1)
        assert 0.15 <= delay_1 <= 0.25  # 0.2 ± 25%

        # Attempt 2: base_delay * 2^2 = 0.1 * 4 = 0.4s (±jitter)
        delay_2 = client_with_retries._calculate_backoff_delay(2)
        assert 0.3 <= delay_2 <= 0.5  # 0.4 ± 25%

    def test_backoff_delay_respects_max_delay(self, settings_with_retries):
        """Test that backoff delay is capped at max_delay."""
        settings = Settings(
            openai_api_key="test-api-key",
            llm_retry_base_delay=1.0,
            llm_retry_max_delay=5.0,
        )
        client = LLMClient(settings)

        # Attempt 10: base_delay * 2^10 = 1024s, but should be capped at 5s
        delay = client._calculate_backoff_delay(10)
        assert delay <= 5.0

    def test_is_retryable_error(self, client_with_retries):
        """Test _is_retryable_error correctly identifies retryable errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        # Retryable errors
        assert client_with_retries._is_retryable_error(
            RateLimitError(message="", response=mock_response, body={})
        )
        assert client_with_retries._is_retryable_error(
            APITimeoutError(request=MagicMock())
        )
        assert client_with_retries._is_retryable_error(
            APIConnectionError(request=MagicMock())
        )
        assert client_with_retries._is_retryable_error(
            InternalServerError(message="", response=mock_response, body={})
        )

        # Non-retryable errors
        mock_response.status_code = 401
        assert not client_with_retries._is_retryable_error(
            AuthenticationError(message="", response=mock_response, body={})
        )
        mock_response.status_code = 400
        assert not client_with_retries._is_retryable_error(
            BadRequestError(message="", response=mock_response, body={})
        )

    def test_retry_logs_warnings(self, client_with_retries, mock_openai_response, caplog):
        """Test that retries are logged as warnings."""
        import logging

        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client

            mock_client.chat.completions.create.side_effect = [
                APITimeoutError(request=MagicMock()),
                mock_openai_response,
            ]

            with patch("time.sleep"):
                with caplog.at_level(logging.WARNING):
                    client_with_retries._call_api("test prompt")

            assert "retrying" in caplog.text

    def test_final_failure_logs_error(self, client_with_retries, caplog):
        """Test that final failure is logged as error."""
        import logging

        with patch.object(client_with_retries, "_client", create=True) as mock_client:
            client_with_retries._client = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 429

            mock_client.chat.completions.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "Rate limit exceeded"}},
            )

            with patch("time.sleep"):
                with caplog.at_level(logging.ERROR):
                    with pytest.raises(LLMRateLimitError):
                        client_with_retries._call_api("test prompt")

            assert "no more retries" in caplog.text
