# Entrypoint Boundaries

Status: Closed

Closed by: current documentation map

## Problem

alpha-os has several entrypoints:

- runtime manifests
- CLI commands
- Python domain APIs
- tests
- scripts

Having several entrypoints is not the problem. The problem is that their roles
are not explicit enough.

This makes it easy to treat the CLI as the core product, or to treat runtime
manifests as the complete research source of truth.

## Current Understanding

The current intended roles are:

- runtime manifests: primary external entrypoint for executable runs
- CLI commands: execution adapter for applying manifests and running reports
- Python domain APIs: core internal entrypoint for tests and application logic
- scripts: legacy or experimental helpers

## Risk

When entrypoint roles are unclear, infrastructure work can look like research
progress.

Examples:

- CLI cleanup can look like improving the core system.
- Manifest expansion can look like capturing research intent.
- Script additions can bypass the manifest and report lifecycle.

## Guard

Do not add a new entrypoint unless it has a clear role that is different from
the existing entrypoints.

Do not promote scripts or CLI commands to the primary path unless they improve a
concrete evaluation workflow.

## Next Decision

When adding a new way to run or inspect alpha-os, decide whether it is a runtime
manifest path, a CLI adapter, a Python API, or a script.

## Close Condition

Close this when the supported entrypoint roles are documented in one place and
new entrypoints can be classified without referring to this issue.

## Later

Revisit this issue when a concrete evaluation is blocked by entrypoint ambiguity.

## Closure Notes

`docs/README.md` now documents the source-of-truth order for current
entrypoints, long-horizon design notes, operational truth, and archive/legacy
context.

The supported entrypoint roles can be classified without this issue:

- root `README.md`: current trusted runtime path and entrypoint commands
- runtime manifests: primary external entrypoint for executable runs
- CLI commands: bounded runtime adapter
- Python APIs: internal application and test entrypoint
- scripts: legacy or experimental helpers unless explicitly promoted
