# Scaling And Migration

## Greenfield Baseline

If alpha-os were started from scratch today, it would be simpler than the
current repository.

The clean baseline would begin with one domain model:

- `hypothesis`
- `prediction`
- `observation`
- `allocation`
- `execution`
- `alpha` as an outcome only

It would not carry:

- `alpha_id` compatibility aliases
- registry-era deployment layers
- legacy runtime fallbacks in the main path

## Current Repo vs Greenfield Target

The current repo is an in-place migration rather than a pure greenfield build.

So the correct architectural standard is not "why does any legacy remain?" but:

- is legacy isolated
- is legacy shrinking
- is legacy prevented from re-entering runtime truth

The runtime should remain:

- hypotheses-first
- persistence-centered on current names and contracts
- protected from research and legacy becoming source of truth

## Multi-Asset Expansion

Multi-asset expansion should not be treated as "add another symbol to the same
BTC-first loop."

The architecture should make explicit:

- observation state by asset
- research scoring by asset
- capital eligibility by asset
- cross-sleeve portfolio allocation

The safe sequence is:

1. keep one reference sleeve stable
2. make sleeve boundaries explicit
3. validate a small number of additional sleeves
4. only then allow broader capital entry

## Thousand-Asset Scaling

Large-scale design requires a different unit of computation and persistence.

The greenfield scaling model should move toward:

- `hypothesis_template`
- `feature_binding`
- `sleeve_state`

This avoids duplicating the same idea into many unrelated asset-specific
records.

At larger scale, the system should separate:

- **control plane**
  - templates
  - bindings
  - sleeve state
  - allocation policy
- **data plane**
  - batched feature loading
  - batched signal evaluation
  - batched observation updates

Allocation should eventually decompose into:

1. hypothesis selection within a sleeve
2. sleeve selection within an asset
3. asset allocation within the global portfolio

## Migration Constraint

The current repo should not try to reach large scale by incrementally appending
asset-specific runtime records forever.

The safe path is:

- validate the bounded runtime on a few assets
- preserve asset-aware abstractions
- avoid new bespoke per-asset assumptions
- document the future template/binding split before broad expansion
