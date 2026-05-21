# Trading Term Drift Boundary

## Problem

Some glossary entries are starting to read like implementation notes rather
than project terminology.

Examples to watch:

- `universe` can mention alpha-os usage, but should not collapse into
  "whatever the current `subject_set` table stores."
- `execution` should describe the domain concept before discussing current
  backtest mechanics.
- `benchmark` can describe how alpha-os uses comparison references, but should
  not become a dumping ground for class-name restrictions.

## Why It Matters

The glossary should remain useful as the alpha-os domain language, not a
generic trading dictionary.

But when transient schema names, migration notes, or implementation cautions
are embedded directly in definitions, readers cannot tell which parts describe
the concept and which parts describe today's implementation.

## Boundary

Glossary entries may include alpha-os-specific meaning when that meaning is
part of the domain language.

Short clarifying notes are fine. Longer naming guards, schema migration
concerns, and implementation cautions should be separated from the core
definition or moved to a boundary/design note.

## Acceptance Criteria

- Glossary entries still explain alpha-os domain usage, not just generic
  trading definitions.
- Core definitions are not primarily phrased in terms of transient table names,
  class names, migration plans, or current engine limitations.
- Project-specific cautions stay short when they appear in the glossary.
- Longer naming or implementation concerns move to a boundary/design note.

## Review Tracker

Track only terms that are likely to drift into implementation notes or overlap
with nearby alpha-os concepts. Add terms as they become relevant.

- [x] `universe`
- [x] `tradable universe`
- [x] `evaluation universe`
- [x] `subject`
- [x] `execution`
- [x] `execution kind`
- [x] `strategy execution`
- [x] `strategy execution kind`
- [x] `run policy`
- [x] `strategy run mode`
- [ ] `portfolio`
- [ ] `backtest`
- [ ] `observable`
- [ ] `feature`
- [ ] `representation`
- [ ] `signal`
- [ ] `signal expression`
- [ ] `signal expression language`
- [ ] `signal family`
- [ ] `signal discovery`
- [ ] `thesis`
- [ ] `research spec`
- [ ] `experiment plan`
- [ ] `execution record`
- [ ] `prediction`
- [ ] `signal contribution`
- [ ] `belief`
- [ ] `sizing method`
- [ ] `trading strategy`
- [ ] `strategy scope`
- [ ] `strategy requirements`
- [ ] `position rule`
- [ ] `portfolio policy`
- [ ] `selection policy`
- [ ] `sizing policy`
- [ ] `sizing engine`
- [ ] `rebalance policy`
- [ ] `risk policy`
- [ ] `rebalance friction policy`
- [ ] `execution policy`
- [ ] `strategy`
- [ ] `strategy spec`
- [x] `strategy run request`
- [x] `strategy run spec`
- [x] `strict OOS run inputs`
- [x] `checkpoint evaluation inputs`
- [x] `paper run inputs`
- [x] `live run inputs`
- [ ] `strict OOS evaluation`
- [ ] `evaluation spec`
- [ ] `evaluation case`
- [ ] `data input`
- [ ] `data source`
- [ ] `strategy checkpoint`
- [ ] `signal train`
- [ ] `train artifact`
- [ ] `checkpoint-based evaluation`
- [ ] `evaluation result`
- [ ] `evaluation run result`
- [ ] `evaluation metric group`
- [ ] `evaluation metric`
- [ ] `evaluation metric group result`
- [ ] `evaluation metric group name`
- [x] `evaluation profile`
- [ ] `benchmark`
- [ ] `net return`
- [ ] `drawdown`
- [ ] `Sharpe ratio`
- [ ] `turnover`
- [ ] `alpha`

## Review Notes

### `universe`

The glossary definition is acceptable as a broad parent term. It should remain
available when the intended meaning is simply "the relevant set of instruments,
assets, securities, or markets."

Code review found one overly broad message, `strategy subject_set or universe
axis`, where the implementation only requires `strategy subject_set`. Fix that
in place instead of creating a separate issue.

`universe_policy` is different: it is a widely used persisted/manifest field for
cross-instrument subject-set assumptions such as base currency, trading
calendar, and benchmark. Renaming or reshaping it should be handled as a
separate boundary task if it becomes painful. See
[`universe-policy-naming-boundary.md`](./universe-policy-naming-boundary.md).

### `tradable universe`

The glossary definition should stay focused on instruments a strategy may hold
or trade. Avoid broad notes about all strategy-observed data because strategy
inputs can include features, macro context, benchmarks, metadata, and other
non-instrument inputs.

Code review found that `tradable universe` is not first-class in code and is
usually represented through `SubjectSet`, which also carries observation and
metadata context. See
[`tradable-universe-code-boundary.md`](./tradable-universe-code-boundary.md).

### `evaluation universe`

The glossary definition is acceptable: an evaluation universe is the set of
instruments included in a specific evaluation run.

Code review found that evaluation runs do not have a first-class
`EvaluationUniverse` model or an evaluation-universe field on `EvaluationSpec`.
The current effective subject set is derived from strategy state. The removed
checkpoint path used checkpoint-owned subject-set metadata. See
[`evaluation-universe-code-boundary.md`](./evaluation-universe-code-boundary.md).

### `subject`

The glossary definition should keep `subject` tied to portfolio-weight-bearing
allocation units. It may cover assets, instruments, baskets, sleeves, pairs, or
spreads, but should not expand to every possible prediction or evaluation
target. Regime or macro-state targets should remain outside `subject` unless
they become allocation targets.

Code review found that `subject_id` appears across prediction, evaluation,
screening, belief synthesis, portfolio construction, and execution paths. See
[`subject-id-code-boundary.md`](./subject-id-code-boundary.md).

### `execution`

The glossary definition is acceptable for order/trade execution, but code and
docs use `execution` across several layers: order execution, portfolio
transition, strategy-run wording, old `execution_kind` behavior, and evaluation
execution ranges. Avoid bare `execution` when a scoped term is
available. See
[`execution-term-boundary.md`](./execution-term-boundary.md).

### `execution kind` / `strategy execution kind`

Do not keep either as a glossary term. These names imply a subtype of
`execution`, but the old `execution_kind` field was not about order execution.

`execution_kind` has been removed rather than renamed as a domain term.
See [`execution-kind-removal.md`](./execution-kind-removal.md).

### `strategy execution`

Do not keep `strategy execution` as a glossary term. It overlaps too easily with
order/trade `execution`.

Use `strategy run` for the broad concept of running a strategy through an engine
context, and keep `execution` reserved for order/trade execution.

### `run policy` / `strategy run mode`

Do not keep either as a glossary term. `run policy` has no first-class
implementation, and `strategy run mode` mostly duplicated the removed
`run_mode` field.

Treat concrete evaluation job shapes and their required inputs directly. Do not
reintroduce `run_mode` as a domain term to rename. See
[`run-mode-removal.md`](./run-mode-removal.md).
