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
- Use the README for current setup and full verification commands.
