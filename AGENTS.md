# alpha-os Agent Guide

## Project

`alpha_os` is the maintained package for signal discovery, strategy definition,
portfolio decisions, and out-of-sample evaluation.

## Structure

```
src/alpha_os/                Mainline package
config/runtime_manifests/    Example runtime manifests
tests/                       Pytest suite
data/                        Local runtime data (gitignored)
```

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

- Prefer maintained work in `src/alpha_os/`.
- Keep changes aligned with signal discovery, strategy specs, evaluation
  protocols, evaluation cases, and evaluation reports.
- Do not commit local runtime data, credentials, logs, or machine-specific
  deployment notes.
- Keep machine-specific operations in `AGENTS.override.md`, which is ignored.

## Testing

- Run focused tests for narrow changes.
- Run `pytest tests/` and `ruff check src tests` before release-oriented
  changes.
