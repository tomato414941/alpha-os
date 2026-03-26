# Future Options Roadmap

This file is a holding area for future options that are not part of the
current working plan.

It is intentionally not the source of truth for:

- the current runtime shape
- current CLI entrypoints
- current operating boundaries
- the current implementation order

Prefer these documents first:

- `README.md`
  - current runtime path
- `OPERATING_BOUNDARIES.md`
  - trusted vs untrusted boundary
- `plan.md`
  - what to do next
- `DESIGN.md`
  - greenfield and long-horizon architecture

## How To Read This File

Items remain here only when they are all of the following:

- future-facing
- optional
- not yet promoted into active implementation

When an item becomes active work, move it to `plan.md`.
When an item becomes part of the architectural baseline, move it to
`DESIGN.md`.

This file should stay short. It is a future-options list, not a delivery
roadmap.

## Future Option Buckets

### 1. Multi-Timeframe Predictive Stack

Possible future direction:

- add tactical intraday sleeves on top of the bounded daily runtime
- add event-aware gating for faster-moving information
- add execution-sensitive predictors where microstructure matters

This is not current runtime work. The current trusted path remains the bounded
hypotheses-first sleeve runtime.

### 2. Cross-Sectional And Relative-Value Sleeves

Possible future direction:

- relative-value prediction across assets
- pair or basket ranking sleeves
- cross-asset allocation that is not just per-sleeve independence

This requires a different runtime contract from the current sleeve-per-asset
directional flow.

### 3. Options And Volatility Intelligence

Possible future direction:

- implied-volatility and skew-aware predictors
- vol-surface regime sleeves
- options-aware risk scaling for directional sleeves

This stays exploratory until the current template-aware control plane is more
stable.

### 4. Cross-Exchange And Lead-Lag Intelligence

Possible future direction:

- multi-exchange flow comparison
- exchange-specific lead-lag signals
- venue disagreement as a predictive input

This should not be treated as near-term implementation until the current
single-venue bounded runtime is trusted for longer.

### 5. Broader Target-Centric Predictive OS

Possible future direction:

- move beyond one market target into multiple target classes
- treat hypothesis health, allocation, execution quality, and data trust as
  first-class prediction targets
- move from hypothesis-centric runtime thinking toward target-centric control
  planes

The concrete greenfield direction for this already lives in `plan.md` and
`DESIGN.md`. This bucket remains here only as a reminder of the broader future
scope.

## Promotion Rule

Move an item out of this file when one of these becomes true:

1. it is now part of the current working queue
2. it now constrains architecture decisions
3. it has been rejected and no longer deserves exploration space

If none of those are true, it can stay here as a future option.
