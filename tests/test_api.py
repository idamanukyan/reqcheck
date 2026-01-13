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
