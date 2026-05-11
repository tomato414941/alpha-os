# Package Manager Boundary

Status: Closed

Closed by: current README and CI setup

## Problem

alpha-os currently documents setup with `pip install -e ".[dev]"`.

The project may want to use `uv` for faster, more reproducible local setup and
CI, but that decision is not captured yet.

## Current Decision

Do not migrate package management yet.

Keep the current `pip`-based setup until a small migration plan is chosen.

## Risk

If package manager choices remain implicit, setup instructions, CI, and local
agent workflows can drift.

If `uv` is introduced ad hoc, the repository may end up supporting two setup
paths without a clear source of truth.

## Guard

Do not mix `uv` and `pip` instructions in primary docs unless the intended
source of truth is explicit.

## Next Decision

When README, CI, or agent setup instructions are next changed, decide whether
`pip install -e ".[dev]"` remains the source of truth or `uv` becomes the source
of truth.

## Close Condition

Close this when README, CI, and agent setup instructions all point to the same
package manager path.

## Later

Evaluate a small `uv` migration:

- choose whether `uv.lock` should be committed
- update README setup commands
- update CI install commands
- confirm `ruff`, `pytest`, and the golden path still pass

## Closure Notes

The current source of truth remains `pip`.

- `README.md` documents `python3 -m venv .venv` followed by
  `pip install -e ".[dev]"`.
- `.github/workflows/ci.yml` installs with
  `python -m pip install -e ".[dev]"`.
- `AGENTS.md` points agents to the README for current setup and verification
  commands.

There is no committed `uv.lock` or competing primary setup path, so setup,
CI, and agent guidance are aligned.
