# reqcheck

> AI-powered requirements quality checker for user stories, tickets, and product specifications.

Bad requirements cost engineering teams more than bad code. `reqcheck` runs your user stories through an LLM-driven quality pipeline that flags ambiguity, missing acceptance criteria, untestable language, and conflicting requirements — before they reach a sprint.

## Why this exists

Most engineering teams catch bad requirements during sprint planning, code review, or — worst case — production. `reqcheck` shifts the catch left, into the writing phase. Product managers get instant feedback on their drafts; engineering leads get a queue of pre-validated stories instead of an inbox of vague tickets.

## What it checks

- **Ambiguity** — vague terms ("user-friendly", "fast", "intuitive") flagged with concrete rewrite suggestions
- **Missing acceptance criteria** — stories without measurable done conditions
- **Untestable language** — requirements no QA engineer can verify
- **Internal conflicts** — contradictions across multiple stories in the same epic
- **Format compliance** — adherence to your team's user story template

## Stack

- **Python** — core logic
- **LLM provider** — pluggable, Claude or OpenAI
- **License** — MIT

## Installation

```bash
git clone https://github.com/idamanukyan/reqcheck.git
cd reqcheck
pip install -r requirements.txt
```

## Configuration

Set your LLM provider and API key:

```bash
export REQCHECK_PROVIDER=claude        # or "openai"
export ANTHROPIC_API_KEY=your-key      # or OPENAI_API_KEY
```

## Usage

Check a single story:

```bash
python -m reqcheck check path/to/story.md
```

Check a directory of stories with JSON output:

```bash
python -m reqcheck check ./tickets --format=json
```

Sample output:

story-042.md
⚠ ambiguity: "the system should be fast" — define a measurable threshold
✗ missing: no acceptance criteria found
⚠ untestable: "users should feel confident" — rewrite as a behavior
story-043.md
✓ passes all checks

## Roadmap

- [ ] Jira integration — pull tickets, post comments with findings
- [ ] Linear / Notion connectors
- [ ] Custom rule sets per team
- [ ] Severity scoring and CI gating

## Contributing

Issues and PRs welcome. MIT-licensed.
