# Lightweight Hypothesis Evaluation Path

## Problem

alpha-os has hypothesis cards, checked-in experiment data, and formal
evaluation CLI commands, but no lightweight path to quickly test a hypothesis
against checked-in data before promoting it into runtime manifests.

## Risk

Early investment research is pushed into the heavier formal evaluation path.

That makes infrastructure work happen before the market hypothesis is judged,
and it increases the chance that alpha-os optimizes the evaluation framework
instead of testing whether a strategy idea has value.

## Guard

Before adding a first-pass strategy idea to `src/alpha_os/` or a runtime
manifest, check whether it can be evaluated as a small experiment under
`experiments/`.

## Next Decision

For `crypto_regime_momentum`, keep the smallest experiment-level evaluator next
to the hypothesis record. It should read the checked-in BTC/ETH dataset, compute
the documented baseline and candidate, and print a compact comparison table.

## Close Condition

Close this when the first lightweight evaluator has produced enough evidence to
decide whether to promote the hypothesis into the formal alpha-os evaluation
path.

## Later

Promote only promising experiment results into the formal alpha-os evaluation
path.
