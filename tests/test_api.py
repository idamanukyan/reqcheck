"""Tests for the REST API."""

import pytest
from fastapi.testclient import TestClient

from reqcheck.api import app
from reqcheck.core.config import Settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""

    def test_analyze_valid_requirement(self, client):
        response = client.post(
            "/analyze",
            json={
                "title": "User Login",
                "description": "Users can log in to the system",
                "acceptance_criteria": ["User enters credentials"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "requirement_id" in data
        assert "issues" in data
        assert "scores" in data

    def test_analyze_missing_title(self, client):
        response = client.post(
            "/analyze",
            json={"description": "No title provided"},
        )
        assert response.status_code == 422  # Validation error

    def test_analyze_empty_title(self, client):
        response = client.post(
            "/analyze",
            json={"title": "", "description": "Empty title"},
        )
        assert response.status_code == 422

    def test_analyze_returns_scores(self, client):
        response = client.post(
            "/analyze",
            json={
                "title": "Test Feature",
                "description": "A test feature description",
                "acceptance_criteria": ["Feature works"],
            },
        )
        data = response.json()
        scores = data["scores"]
        assert 0 <= scores["ambiguity"] <= 1
        assert 0 <= scores["completeness"] <= 1
        assert 0 <= scores["testability"] <= 1


class TestMarkdownEndpoint:
    """Tests for /analyze/markdown endpoint."""

    def test_markdown_returns_text(self, client):
        response = client.post(
            "/analyze/markdown",
            json={
                "title": "Test",
                "description": "Test description",
                "acceptance_criteria": ["Test AC"],
            },
        )
        assert response.status_code == 200
        assert "# QA Analysis" in response.text


class TestSummaryEndpoint:
    """Tests for /analyze/summary endpoint."""

    def test_summary_returns_text(self, client):
        response = client.post(
            "/analyze/summary",
            json={
                "title": "Test",
                "description": "Test description",
                "acceptance_criteria": ["Test AC"],
            },
        )
        assert response.status_code == 200
        assert "Test" in response.text


class TestBatchEndpoint:
    """Tests for /analyze/batch endpoint."""

    def test_batch_analyze(self, client):
        response = client.post(
            "/analyze/batch",
            json={
                "requirements": [
                    {"title": "Req 1", "description": "First"},
                    {"title": "Req 2", "description": "Second"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_list(self, client):
        response = client.post(
            "/analyze/batch",
            json={"requirements": []},
        )
        assert response.status_code == 422  # Validation: min 1 required


class TestAPIKeyAuthentication:
    """Tests for API key authentication."""

    @pytest.fixture
    def auth_client(self):
        """Create client with auth-enabled settings."""
        from reqcheck.api import create_app
        from reqcheck.core.config import reset_settings

        reset_settings()
        auth_settings = Settings(
            auth_enabled=True,
            api_keys="test-key-1,test-key-2",
            enable_llm_analysis=False,
        )
        app = create_app(auth_settings)
        return TestClient(app)

    def test_request_without_api_key_fails_when_auth_enabled(self, auth_client):
        """Test that requests without API key are rejected when auth is enabled."""
        response = auth_client.post(
            "/analyze",
            json={
                "title": "Test",
                "description": "Test",
                "acceptance_criteria": ["AC1"],
            },
        )
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_request_with_invalid_api_key_fails(self, auth_client):
        """Test that invalid API keys are rejected."""
        response = auth_client.post(
            "/analyze",
            json={
                "title": "Test",
                "description": "Test",
                "acceptance_criteria": ["AC1"],
            },
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]

    def test_request_with_valid_api_key_succeeds(self, auth_client):
        """Test that valid API keys are accepted."""
        response = auth_client.post(
            "/analyze",
            json={
                "title": "Test",
                "description": "Test description",
                "acceptance_criteria": ["AC1"],
            },
            headers={"X-API-Key": "test-key-1"},
        )
        assert response.status_code == 200

    def test_health_endpoint_accessible_without_auth(self, auth_client):
        """Test that health endpoint doesn't require authentication."""
        response = auth_client.get("/health")
        assert response.status_code == 200

    def test_auth_disabled_allows_all_requests(self):
        """Test that disabled auth allows requests without API key."""
        from reqcheck.api import create_app
        from reqcheck.core.config import reset_settings

        reset_settings()
        no_auth_settings = Settings(
            auth_enabled=False,
            enable_llm_analysis=False,
        )
        app = create_app(no_auth_settings)
        client = TestClient(app)

        response = client.post(
            "/analyze",
            json={
                "title": "Test",
                "description": "Test",
                "acceptance_criteria": ["AC1"],
            },
        )
        assert response.status_code == 200


class TestRequestSizeLimits:
    """Tests for request size validation."""

    @pytest.fixture
    def strict_client(self):
        """Create client with strict size limits."""
        from reqcheck.api import create_app
        from reqcheck.core.config import reset_settings

        reset_settings()
        strict_settings = Settings(
            max_title_length=50,
            max_description_length=100,
            max_acceptance_criteria_count=3,
            max_acceptance_criteria_length=50,
            max_metadata_size_bytes=256,  # Minimum allowed value
            enable_llm_analysis=False,
        )
        app = create_app(strict_settings)
        return TestClient(app)

    def test_title_exceeds_max_length(self, strict_client):
        """Test rejection of title exceeding max length."""
        response = strict_client.post(
            "/analyze",
            json={
                "title": "x" * 100,  # Exceeds 50 char limit
                "description": "Test",
                "acceptance_criteria": ["AC1"],
            },
        )
        assert response.status_code == 422
        assert "Title exceeds" in str(response.json())

    def test_description_exceeds_max_length(self, strict_client):
        """Test rejection of description exceeding max length."""
        response = strict_client.post(
            "/analyze",
            json={
                "title": "Test",
                "description": "x" * 200,  # Exceeds 100 char limit
                "acceptance_criteria": ["AC1"],
            },
        )
        assert response.status_code == 422
        assert "Description exceeds" in str(response.json())

    def test_too_many_acceptance_criteria(self, strict_client):
        """Test rejection of too many acceptance criteria."""
        response = strict_client.post(
            "/analyze",
            json={
                "title": "Test",
                "description": "Test",
                "acceptance_criteria": ["AC1", "AC2", "AC3", "AC4", "AC5"],  # Exceeds 3
            },
        )
        assert response.status_code == 422
        assert "Too many acceptance criteria" in str(response.json())

    def test_acceptance_criteria_too_long(self, strict_client):
        """Test rejection of individual AC exceeding length."""
        response = strict_client.post(
            "/analyze",
            json={
                "title": "Test",
                "description": "Test",
                "acceptance_criteria": ["x" * 100],  # Exceeds 50 char limit
            },
        )
        assert response.status_code == 422
        assert "Acceptance criterion" in str(response.json())

    def test_metadata_exceeds_size_limit(self, strict_client):
        """Test rejection of metadata exceeding size limit."""
        response = strict_client.post(
            "/analyze",
            json={
                "title": "Test",
                "description": "Test",
                "acceptance_criteria": ["AC1"],
                "metadata": {"large_field": "x" * 300},  # Exceeds 256 bytes
            },
        )
        assert response.status_code == 422
        assert "Metadata exceeds" in str(response.json())

    def test_valid_request_within_limits(self, strict_client):
        """Test that requests within limits are accepted."""
        response = strict_client.post(
            "/analyze",
            json={
                "title": "Short title",
                "description": "Short desc",
                "acceptance_criteria": ["AC1", "AC2"],
                "metadata": {"key": "val"},
            },
        )
        assert response.status_code == 200

    def test_batch_validates_all_requirements(self, strict_client):
        """Test that batch endpoint validates all requirements."""
        response = strict_client.post(
            "/analyze/batch",
            json={
                "requirements": [
                    {"title": "Valid", "description": "OK", "acceptance_criteria": ["AC1"]},
                    {
                        "title": "x" * 100,
                        "description": "Invalid",
                        "acceptance_criteria": ["AC1"],
                    },
                ]
            },
        )
        assert response.status_code == 422


class TestCORSConfiguration:
    """Tests for CORS configuration."""

    def test_cors_preflight_request(self, client):
        """Test CORS preflight handling."""
        response = client.options(
            "/analyze",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-API-Key",
            },
        )
        assert response.status_code in [200, 204]
