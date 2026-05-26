# alpha-os Agent Guide

## Structure

Prefer a modular monolith with vertical slices and AI-readable boundaries.
Place new code near the workflow it serves instead of creating broad abstract
layers by default.

Do not introduce manifest DSLs or large layered architecture unless there is a
clear concrete need. Split shared directories only after multiple concrete
implementations need a shared home.

## Conventions

- Get explicit approval before keeping branches, aliases, or deprecated
  interfaces for backward compatibility.
- Do not commit local runtime data, credentials, logs, or machine-specific
  deployment notes.

## Testing

- Use `uv` for project commands.
