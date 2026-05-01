# alpha-os Agent Guide

## Project

`alpha_os` supports:

- lightweight investment hypothesis research
- signal, strategy-candidate, and portfolio-construction generation, screening,
  and constrained optimization
- strategy candidate backtesting and OOS evaluation
- promotion, rejection, and baseline decision records

## Structure

Prefer a modular monolith with vertical slices and AI-readable boundaries.
Place new code near the workflow it serves instead of creating broad abstract
layers by default.

Do not introduce manifest DSLs or large layered architecture unless there is a
clear concrete need. Split shared directories only after multiple concrete
implementations need a shared home.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/
ruff check src tests
python -m alpha_os --help
```

Install the optional data-service integration only when a compatible
`signal-noise` package is available:

```bash
pip install -e ".[data]"
```

## Conventions

- Prefer maintained work in the main package unless the task is explicitly a
  lightweight experiment, documentation change, or local operations note.
- Keep changes aligned with investment hypotheses, signal and strategy-candidate
  generation, portfolio-construction optimization, backtests, OOS evaluation,
  screening, and decision records.
- Do not commit local runtime data, credentials, logs, or machine-specific
  deployment notes.
- Keep machine-specific operations in `AGENTS.override.md`, which is ignored.

## Testing

- Run focused tests for narrow changes.
- Run `pytest tests/` and `ruff check src tests` before release-oriented
  changes.
