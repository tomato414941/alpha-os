# Target Term Boundary

Status: Closed

Closed by: `EvaluationSpec.target_ids` and the unused `TargetDefinition`
registry were removed from the active package. Predictor evaluation can
introduce prediction-target terminology later in its own boundary.

## Problem

`target` is too broad in a project that mixes ML, trading, portfolio
construction, and evaluation.

Without alpha-os context, `target` can mean:

- prediction target or ML label
- evaluation outcome
- portfolio target weight or target position
- target volatility or exposure target
- policy objective or reward

Alpha OS currently uses `target_id` mostly for prediction target definitions
such as `residual_return_3d`, but the shorter word `target` appears in nearby
portfolio and evaluation concepts with different meanings.

## Risk

If `target` remains a standalone project term, readers cannot tell whether a
field is describing:

- what a signal or belief predicts
- what an evaluation measures
- what a portfolio allocator is trying to hold
- what a policy is optimizing

This can make `strategy.target_id`, `default_target_id`, and evaluation
`target_id` look interchangeable even when they may represent different
responsibilities.

## Boundary

Use `prediction target` for the current `target_id` concept when it refers to a
labeled outcome definition such as `residual_return_3d`.

Do not add `target` as a standalone glossary term.

Keep portfolio meanings explicit with terms such as target weight, target
volatility, target exposure, or target position.

## Current Suspects

- `TradingStrategySpec.target_id`
- `EvaluationRunRequest.default_target_id`
- `TargetDefinition`
- documentation that says `target` without a qualifier

## Close Condition

Close this when `target` has been disambiguated in glossary, docs, and code
boundaries, or when remaining standalone uses are explicitly justified by local
context.
