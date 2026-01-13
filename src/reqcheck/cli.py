"""Command-line interface for reqcheck."""

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel

from reqcheck.core.analyzer import RequirementsAnalyzer
from reqcheck.core.config import Settings, get_settings
from reqcheck.core.models import Requirement
from reqcheck.output.formatters import (
    format_checklist,
    format_json,
    format_markdown,
    format_summary,
)

console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="reqcheck")
def main():
    """AI-powered requirements quality checker."""
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True), required=False)
@click.option(
    "--stdin",
    is_flag=True,
    help="Read requirement from stdin as JSON",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "markdown", "summary", "checklist"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file (default: stdout)",
)
@click.option(
    "--min-severity",
    type=click.Choice(["blocker", "warning", "suggestion"]),
    default="suggestion",
    help="Minimum severity to report",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Disable LLM analysis (rule-based only)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def analyze(
    input_file: str | None,
    stdin: bool,
    output_format: str,
    output: str | None,
    min_severity: str,
    no_llm: bool,
    verbose: bool,
):
    """
    Analyze a requirement for quality issues.

    INPUT_FILE: Path to JSON file containing the requirement.
    Use --stdin to read from standard input instead.

    Example JSON format:
    {
        "title": "User login",
        "description": "Users should be able to log in",
        "acceptance_criteria": ["User can enter email and password"]
    }
    """
    setup_logging(verbose)

    # Read input
    if stdin:
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON input: {e}[/red]")
            sys.exit(1)
    elif input_file:
        try:
            with open(input_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in {input_file}: {e}[/red]")
            sys.exit(1)
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {input_file}[/red]")
            sys.exit(1)
    else:
        console.print("[red]Error: Provide INPUT_FILE or use --stdin[/red]")
        sys.exit(1)

    # Parse requirement
    try:
        requirement = Requirement(**data)
    except Exception as e:
        console.print(f"[red]Error: Invalid requirement format: {e}[/red]")
        sys.exit(1)

    # Configure settings
    settings = get_settings()
    if no_llm:
        settings.enable_llm_analysis = False
    settings.min_severity = min_severity

    # Run analysis
    with console.status("[bold green]Analyzing requirement..."):
        analyzer = RequirementsAnalyzer(settings)
        report = analyzer.analyze(requirement)

    # Format output
    if output_format == "json":
        result = format_json(report)
    elif output_format == "markdown":
        result = format_markdown(report, settings)
    elif output_format == "summary":
        result = format_summary(report)
    elif output_format == "checklist":
        result = format_checklist(report)
    else:
        result = format_summary(report)

    # Write output
    if output:
        Path(output).write_text(result)
        console.print(f"[green]Report written to {output}[/green]")
    else:
        if output_format == "markdown":
            console.print(Markdown(result))
        else:
            console.print(result)

    # Exit code based on blockers
    if report.blocker_count > 0:
        sys.exit(1)


@main.command()
@click.option(
    "--title",
    "-t",
    required=True,
    help="Requirement title",
)
@click.option(
    "--description",
    "-d",
    default="",
    help="Requirement description",
)
@click.option(
    "--ac",
    "acceptance_criteria",
    multiple=True,
    help="Acceptance criterion (can be specified multiple times)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "markdown", "summary", "checklist"]),
    default="summary",
    help="Output format",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Disable LLM analysis",
)
def quick(
    title: str,
    description: str,
    acceptance_criteria: tuple[str, ...],
    output_format: str,
    no_llm: bool,
):
    """
    Quick analysis from command line arguments.

    Example:
        reqcheck quick -t "User login" -d "Users can log in" --ac "Can enter email" --ac "Can enter password"
    """
    requirement = Requirement(
        title=title,
        description=description,
        acceptance_criteria=list(acceptance_criteria),
    )

    settings = get_settings()
    if no_llm:
        settings.enable_llm_analysis = False

    with console.status("[bold green]Analyzing..."):
        analyzer = RequirementsAnalyzer(settings)
        report = analyzer.analyze(requirement)

    # Format and display
    if output_format == "json":
        console.print(format_json(report))
    elif output_format == "markdown":
        console.print(Markdown(format_markdown(report, settings)))
    elif output_format == "checklist":
        console.print(format_checklist(report))
    else:
        console.print(format_summary(report))

    if report.blocker_count > 0:
        sys.exit(1)


@main.command()
def config():
    """Show current configuration."""
    settings = get_settings()

    console.print(Panel.fit("[bold]reqcheck Configuration[/bold]"))
    console.print()

    # API Configuration
    api_key_status = "[green]Set[/green]" if settings.openai_api_key else "[red]Not set[/red]"
    console.print(f"OpenAI API Key: {api_key_status}")
    console.print(f"OpenAI Model: {settings.openai_model}")
    console.print(f"LLM Analysis: {'Enabled' if settings.enable_llm_analysis else 'Disabled'}")
    console.print(f"Rule-based Analysis: {'Enabled' if settings.enable_rule_based_analysis else 'Disabled'}")
    console.print()

    # Output Configuration
    console.print(f"Default Output Format: {settings.default_output_format}")
    console.print(f"Minimum Severity: {settings.min_severity}")
    console.print()

    # API Server
    console.print(f"API Server: {settings.api_host}:{settings.api_port}")


@main.command()
def demo():
    """Run analysis on a demo requirement."""
    demo_requirement = Requirement(
        title="User Password Reset",
        description="Users should be able to reset their password when they forget it. The system should handle this appropriately and securely.",
        acceptance_criteria=[
            "User can request password reset",
            "System sends email quickly",
            "Password reset link works correctly",
            "New password is validated properly",
        ],
        type="story",
    )

    console.print(Panel.fit("[bold]Demo: Analyzing a sample requirement[/bold]"))
    console.print()

    console.print("[bold]Input Requirement:[/bold]")
    console.print(f"  Title: {demo_requirement.title}")
    console.print(f"  Description: {demo_requirement.description}")
    console.print("  Acceptance Criteria:")
    for ac in demo_requirement.acceptance_criteria:
        console.print(f"    - {ac}")
    console.print()

    settings = get_settings()
    # Force rule-based only for demo
    settings.enable_llm_analysis = False

    with console.status("[bold green]Running analysis..."):
        analyzer = RequirementsAnalyzer(settings)
        report = analyzer.analyze(demo_requirement)

    console.print(Markdown(format_markdown(report, settings)))


if __name__ == "__main__":
    main()
