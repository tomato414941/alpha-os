# Signal Noise Ingestion Boundary

## Problem

alpha-os has signal-noise backfill and evaluation input generation helpers, but
their status is unclear.

Examples:

- `signal_noise_backfill`
- `generate_evaluation_inputs_from_signal_noise`

## Risk

These helpers may accidentally become a second data ingestion system inside
alpha-os.

That would blur the intended boundary:

```text
signal-noise provides observation data
alpha-os evaluates investment hypotheses
```

## Decision Needed

Classify each signal-noise ingestion helper as either:

- a supported alpha-os import workflow
- an internal temporary ingestion helper

## Close Condition

Close this when each signal-noise ingestion helper has an explicit status and
owner boundary.

## Not Now

Do not add collector-name mappings or exchange-specific fetchers to alpha-os to
work around missing signal-noise observation mappings.

