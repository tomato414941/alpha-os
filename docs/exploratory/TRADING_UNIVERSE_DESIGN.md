# Trading Universe Design

Exploratory design note.

This file describes future options for how the system might decide which assets
or sleeves deserve active capital. It is not the current trusted runtime shape,
current sleeve set, or current scheduler contract.

Prefer:

- `README.md` for the current bounded runtime path
- `OPERATING_BOUNDARIES.md` for the current operating boundary
- `DESIGN.md` for the long-horizon architecture
- `plan.md` for active implementation order

## Core Question

The future question is not only "which hypotheses should we trade?" but also:

- which assets should have active sleeves
- which sleeves should graduate from observation to capital
- which execution venues are worth serving

This is broader than the current bounded `BTC` reference sleeve plus `ETH`
validation sleeve.

## Universe Layers

Future universe design will likely need to separate at least four layers:

| Layer | Purpose | Example |
|------|---------|---------|
| research universe | where ideas are explored broadly | many assets, many features |
| validation universe | where sleeves prove they generalize | small multi-asset set |
| capital universe | where sleeves are allowed to receive budget | bounded trusted set |
| execution universe | where actual orders can be placed | venue- and liquidity-constrained |

The current runtime only partially separates these layers.

## Future Options

### 1. Template-Coverage-Driven Sleeve Expansion

Possible direction:

- open new sleeves where template gaps remain large
- keep bounded reference sleeves stable
- promote new sleeves only after serious coverage and capital-path evidence

### 2. Signal-Support-Driven Asset Selection

Possible direction:

- prefer assets with stronger feature support
- require enough data quality and actionability before adding budget
- avoid opening sleeves where the runtime cannot support the target ideas

### 3. Predictive-Power-Driven Capital Universe

Possible direction:

- use sleeve-level live quality, actionability, and contribution
- prefer assets where hypotheses repeatedly convert into backed sleeves
- remove sleeves that remain observational but never become capital-relevant

### 4. Venue- and Liquidity-Constrained Execution Universe

Possible direction:

- separate "interesting to study" from "safe to trade"
- require venue availability, liquidity, and execution quality
- keep execution universe narrower than research universe

## Not Yet Active

This note does not imply that the project should immediately build:

- a global trading-universe module
- a dynamic many-asset allocator
- a cross-sectional execution engine

Those would only make sense after the current template-aware bounded sleeve
runtime is more mature.

## Promotion Rule

Move material out of this file when it becomes either:

1. current implementation work
2. architectural baseline

If it is neither, it should stay exploratory.
