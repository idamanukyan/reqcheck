=====

# reqcheck

> AI-powered requirements quality checker for user stories, tickets, and product specifications.

Bad requirements cost engineering teams more than bad code. `reqcheck` runs your user stories through an LLM-driven quality pipeline that flags ambiguity, missing acceptance criteria, untestable language, and conflicting requirements — before they reach a sprint.

## Why this exists

Most engineering teams catch bad requirements during sprint planning, code review, or — worst case — production. `reqcheck` shifts the catch left, into the writing phase. Product managers get instant feedback on their drafts; engineering leads get a queue of pre-validated stories instead of an inbox of vague tickets.

## What it checks

- **Ambiguity** — vague terms ("user-friendly," "fast," "intuitive") flagged with concrete rewrite suggestions
- **Missing acceptance criteria** — stories without measurable done conditions
- **Untestable language** — requirements no QA engineer can verify
- **Internal conflicts** — contradictions across multiple stories in the same epic
- **Format compliance** — adherence to your team's user story template

## Stack

- **Python** — core logic
- **LLM provider** — pluggable (Claude / OpenAI)
- **License** — MIT

## Usage

[FILL: drop in your actual CLI or API example — e.g. `reqcheck path/to/story.md` with sample output]

```bash
# Install
pip install reqcheck

# Run against a single story
reqcheck check story.md

# Run against a directory of stories
reqcheck check ./tickets --format=json
```

## Roadmap

- [ ] Jira integration (pull tickets, post comments with findings)
- [ ] Linear / Notion connectors
- [ ] Custom rule sets per team
- [ ] Severity scoring

## Contributing

Issues and PRs welcome. This project is MIT-licensed.

=====
