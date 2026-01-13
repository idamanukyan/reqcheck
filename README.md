# reqcheck

AI-powered requirements quality checker for user stories, tickets, and specifications.

Catch ambiguity, missing details, and untestable criteria **before** development starts.

## Installation

```bash
pip install reqcheck
```

## Quick Start

```bash
# Run the demo to see it in action
reqcheck demo

# Analyze a requirement file
reqcheck analyze requirement.json

# Quick inline analysis
reqcheck quick -t "User login" -d "Users can log in" --ac "Login works correctly"
```

## Configuration

Set your OpenAI API key for LLM-powered analysis (optional - works without it using rule-based analysis):

```bash
export REQCHECK_OPENAI_API_KEY=your-api-key
```

## Usage

### CLI

```bash
# Analyze from JSON file
reqcheck analyze requirement.json --format markdown

# Quick analysis from arguments
reqcheck quick -t "User Login" -d "Users should be able to log in" --ac "Can enter credentials"

# Output formats: json, markdown, summary, checklist
reqcheck analyze requirement.json --format json

# Show current config
reqcheck config
```

### Python API

```python
from reqcheck import analyze_requirement, Requirement

# From dict
report = analyze_requirement({
    "title": "User Login",
    "description": "Users should be able to log in to the system",
    "acceptance_criteria": ["User can enter email and password"]
})

# From Requirement object
requirement = Requirement(
    title="User Login",
    description="Users should be able to log in to the system",
    acceptance_criteria=["User can enter email and password"]
)
report = analyze_requirement(requirement)

# Check results
print(f"Ready for dev: {report.is_ready_for_dev}")
print(f"Issues: {len(report.issues)}")
print(f"Score: {report.scores.overall:.0%}")
```

### REST API

```bash
# Start the server
python -m reqcheck.api

# Analyze a requirement
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"title": "User login", "description": "...", "acceptance_criteria": [...]}'

# Get markdown report
curl -X POST http://localhost:8000/analyze/markdown \
  -H "Content-Type: application/json" \
  -d '{"title": "User login", "description": "..."}'

# Batch analysis (up to 10)
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"requirements": [{"title": "..."}, {"title": "..."}]}'
```

## What It Checks

| Category | Examples |
|----------|----------|
| **Ambiguity** | Vague terms ("appropriately", "properly"), passive voice, unclear pronouns |
| **Completeness** | Missing acceptance criteria, no error handling, escape hatches ("etc.") |
| **Testability** | Unmeasurable outcomes, subjective criteria ("user-friendly") |
| **Risk** | Security concerns, payment handling, third-party integrations |

## Example Output

```
✗ User Password Reset

Issues: 0 blockers, 11 warnings, 1 suggestions
Scores: Ambiguity 41% | Completeness 100% | Testability 46%
Overall: 63%

Top Issues:
  [?] Vague term "appropriately" - specify measurable criteria
  [?] Vague quantifier "quickly" - specify exact values or ranges
  [?] Passive voice "is validated" - specify who/what performs this action
```

## JSON Input Format

```json
{
  "title": "User Password Reset",
  "description": "Users should be able to reset their password...",
  "acceptance_criteria": [
    "User can request password reset",
    "System sends reset email within 60 seconds",
    "Reset link expires after 24 hours"
  ],
  "type": "story",
  "metadata": {
    "priority": "high",
    "labels": ["authentication"]
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REQCHECK_OPENAI_API_KEY` | (none) | OpenAI API key for LLM analysis |
| `REQCHECK_OPENAI_MODEL` | gpt-4o-mini | Model to use |
| `REQCHECK_ENABLE_LLM_ANALYSIS` | true | Enable/disable LLM analysis |
| `REQCHECK_MIN_SEVERITY` | suggestion | Minimum severity to report |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
```

## License

MIT
