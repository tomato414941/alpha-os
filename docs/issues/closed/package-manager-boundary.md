# Package Manager Boundary

Status: Closed

Closed by: uv-based README and CI setup

## Problem

alpha-os previously documented setup with `pip install -e ".[dev]"`.

The project may want to use `uv` for faster, more reproducible local setup and
CI, but that decision is not captured yet.

## Current Decision

Use `uv` as the project package-manager path.

## Risk

If package manager choices remain implicit, setup instructions, CI, and local
agent workflows can drift.

If `uv` is introduced ad hoc, the repository may end up supporting two setup
paths without a clear source of truth.

## Guard

Do not mix `uv` and `pip` instructions in primary docs unless the intended
source of truth is explicit.

## Next Decision

When README, CI, or agent setup instructions are next changed, keep them aligned
with the `uv` path unless a new package-manager migration is chosen.

## Close Condition

Close this when README, CI, and agent setup instructions all point to the same
package manager path.

## Later

Revisit only if the project intentionally migrates away from `uv`.

## Closure Notes

The current source of truth is `uv`.

- `uv.lock` is committed.
- `README.md` documents `uv sync --extra dev`.
- `.github/workflows/ci.yml` installs with `uv sync --extra dev --locked`.
- `AGENTS.md` tells agents to use `uv` for project setup and command execution.

Setup, CI, and agent guidance are aligned on one package-manager path.
