# Local Runtime Data Boundary

## Problem

The local `data/` directory mixes runtime state, caches, logs, evaluation
artifacts, and old experiment outputs.

Some paths look especially ambiguous:

- `data/BTC/alpha_registry.db`
- `data/BTC/alpha_registry_l2.db`
- `data/BTC/metrics/`
- `data/BTC/logs/`
- `data/ETH/`
- `data/F1/`
- global macro evaluation databases and reports directly under `data/`

The `data/BTC/` namespace is especially confusing because it can make an old
crypto or testnet runtime database look like the central alpha registry for the
project.

## Current Decision

Do not delete local runtime data yet.

The data may include useful historical state or outputs, and deleting it before
understanding current dependencies would be risky.

## Risk

If local runtime data is treated as a source of truth, old experiment state can
be confused with current research evidence.

If specific local paths are documented in agent instructions, they can look more
official than they are.

## Guard

Do not document specific local runtime database paths in `AGENTS.override.md`
unless the user explicitly asks for that path to be treated as current
operational context.

Prefer describing the rule instead: runtime data under `data/` is local-only and
must not be committed.

## Next Decision

Before deleting, archiving, or using any local `data/` artifact as research
evidence, classify that artifact as fixture, generated artifact, runtime cache,
operational state, log, or temporary output.

## Close Condition

Close this when local `data/` artifacts are either classified by type or removed
from the project workflow.

## Later

Classify local data into at least these groups before cleanup:

- reproducible evaluation fixtures
- generated evaluation artifacts
- runtime caches
- live or testnet operational state
- logs and temporary outputs

Only delete or archive local data after that classification is clear.
