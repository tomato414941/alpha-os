# Strategy Evaluation Path Boundary

## Problem

alpha-os currently runs candidates through different evaluation paths depending
on how the candidate was produced.

Examples:

- hand-written trainless rules use `strategy_backtest`
- discovered signals use signal discovery execution
- frozen discovery artifacts use fixed-state replay

## Risk

The evaluation path can become tied to candidate origin.

That can make the engine structure feel like the research result, even when the
user only wants to know whether a fixed candidate beats a baseline under shared
conditions.

## Boundary

This issue is about engine routing.

It is separate from the archived common strategy comparison contract issue,
which is about the comparison contract that makes candidate results comparable.

## Current Finding

Hand-written trainless candidates and signal-discovery-derived candidates both
feed the decision backtest and report metric machinery.

The problem is not that every metric is totally separate. The problem is that
engine path names still carry conceptual weight.

## Desired Direction

Signal-discovery-derived strategies should eventually flow through the same
strategy backtest boundary as hand-written strategies.

Signal discovery should produce or select strategy state. It should not remain
the reason a strategy uses a separate evaluation path.

## Close Condition

Close this when alpha-os has a clear boundary between candidate production and
the engine path used to evaluate a fixed candidate.
