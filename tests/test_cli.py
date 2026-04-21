"""Tests for the CLI interface."""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from reqcheck.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_requirement_file():
    """Create a temporary JSON file with a sample requirement."""
    data = {
        "title": "User Registration",
        "description": "Users can register with email and password.",
        "acceptance_criteria": [
            "GIVEN a new visitor WHEN they fill the registration form THEN an account is created"
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def poor_requirement_file():
    """Create a temporary JSON file with a poor requirement."""
    data = {
        "title": "Login",
        "description": "Users should be able to log in properly.",
        "acceptance_criteria": [],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def invalid_json_file():
    """Create a temporary file with invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{invalid json")
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestAnalyzeCommand:
    """Tests for the 'analyze' command."""

    def test_analyze_file_markdown(self, runner, sample_requirement_file):
        result = runner.invoke(
            main, ["analyze", sample_requirement_file, "--no-llm", "-f", "markdown"]
        )
        assert result.exit_code == 0

    def test_analyze_file_json(self, runner, sample_requirement_file):
        result = runner.invoke(
            main, ["analyze", sample_requirement_file, "--no-llm", "-f", "json"]
        )
        assert result.exit_code == 0

    def test_analyze_file_summary(self, runner, sample_requirement_file):
        result = runner.invoke(
            main, ["analyze", sample_requirement_file, "--no-llm", "-f", "summary"]
        )
        assert result.exit_code == 0

    def test_analyze_file_checklist(self, runner, sample_requirement_file):
        result = runner.invoke(
            main, ["analyze", sample_requirement_file, "--no-llm", "-f", "checklist"]
        )
        assert result.exit_code == 0

    def test_analyze_poor_requirement_exits_nonzero(self, runner, poor_requirement_file):
        result = runner.invoke(
            main, ["analyze", poor_requirement_file, "--no-llm"]
        )
        assert result.exit_code == 1

    def test_analyze_invalid_json(self, runner, invalid_json_file):
        result = runner.invoke(main, ["analyze", invalid_json_file, "--no-llm"])
        assert result.exit_code == 1

    def test_analyze_no_input(self, runner):
        result = runner.invoke(main, ["analyze"])
        assert result.exit_code != 0

    def test_analyze_stdin(self, runner):
        data = json.dumps({
            "title": "Test",
            "description": "A test requirement for stdin input.",
            "acceptance_criteria": ["GIVEN input WHEN processed THEN output is correct"],
        })
        result = runner.invoke(
            main, ["analyze", "--stdin", "--no-llm", "-f", "summary"], input=data
        )
        assert result.exit_code == 0

    def test_analyze_output_to_file(self, runner, sample_requirement_file):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as out:
            out_path = out.name

        try:
            result = runner.invoke(
                main,
                [
                    "analyze", sample_requirement_file,
                    "--no-llm", "-f", "markdown", "-o", out_path,
                ],
            )
            assert result.exit_code == 0
            content = open(out_path).read()
            assert len(content) > 0
        finally:
            os.unlink(out_path)

    def test_analyze_min_severity_blocker(self, runner, sample_requirement_file):
        result = runner.invoke(
            main,
            ["analyze", sample_requirement_file, "--no-llm", "--min-severity", "blocker"],
        )
        assert result.exit_code == 0

    def test_analyze_verbose(self, runner, sample_requirement_file):
        result = runner.invoke(
            main, ["analyze", sample_requirement_file, "--no-llm", "-v"]
        )
        assert result.exit_code == 0


class TestQuickCommand:
    """Tests for the 'quick' command."""

    def test_quick_basic(self, runner):
        result = runner.invoke(
            main,
            [
                "quick", "-t", "User Login",
                "-d", "Users can log in with email and password",
                "--ac", "GIVEN valid credentials WHEN submitted THEN user is logged in",
                "--no-llm",
            ],
        )
        assert result.exit_code == 0

    def test_quick_with_acceptance_criteria(self, runner):
        result = runner.invoke(
            main,
            [
                "quick",
                "-t", "User Login",
                "-d", "Users can log in with email and password",
                "--ac", "GIVEN valid credentials WHEN submitted THEN user is authenticated",
                "--ac", "GIVEN invalid credentials WHEN submitted THEN error is shown",
                "--no-llm",
            ],
        )
        assert result.exit_code == 0

    def test_quick_json_format(self, runner):
        result = runner.invoke(
            main,
            [
                "quick", "-t", "Test Feature",
                "-d", "Users can test the feature with a form",
                "--ac", "GIVEN a user WHEN they submit THEN result is saved in 2 seconds",
                "--no-llm", "-f", "json",
            ],
        )
        assert result.exit_code == 0

    def test_quick_poor_requirement_exits_nonzero(self, runner):
        result = runner.invoke(
            main,
            ["quick", "-t", "Bad", "-d", "", "--no-llm"],
        )
        assert result.exit_code == 1

    def test_quick_missing_title(self, runner):
        result = runner.invoke(main, ["quick", "--no-llm"])
        assert result.exit_code != 0


class TestDemoCommand:
    """Tests for the 'demo' command."""

    def test_demo_runs(self, runner):
        result = runner.invoke(main, ["demo"])
        assert result.exit_code == 0 or result.exit_code == 1
        # Demo uses a poor requirement, so may exit 0 or 1 depending on blockers
        assert len(result.output) > 50

    def test_demo_produces_output(self, runner):
        result = runner.invoke(main, ["demo"])
        assert len(result.output) > 100


class TestConfigCommand:
    """Tests for the 'config' command."""

    def test_config_shows_settings(self, runner):
        result = runner.invoke(main, ["config"])
        assert result.exit_code == 0
        assert "Model" in result.output or "OpenAI" in result.output

    def test_config_shows_api_key_status(self, runner):
        result = runner.invoke(main, ["config"])
        assert result.exit_code == 0
        assert "Set" in result.output or "Not set" in result.output


class TestVersionFlag:
    """Tests for the --version flag."""

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
