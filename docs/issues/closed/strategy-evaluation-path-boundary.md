# Strategy Evaluation Path Boundary

Status: Closed

Closed by: the old direct, signal-discovery, decision-backtest, and checkpoint
evaluation paths were removed. Future evaluation paths should be designed from
the `TradingStrategy` contract rather than from candidate origin.

## Problem

alpha-os currently runs candidates through different evaluation paths depending
on how the candidate was produced.

Examples:

- direct hand-written paths use resolved backtest input series
- discovered signals use signal discovery execution
- the removed strategy-checkpoint path used checkpoint-based evaluation

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
feed the decision backtest and run result metric machinery.

The problem is not that every metric is totally separate. The problem is that
engine path names still carry conceptual weight.

The previous checkpoint-based evaluation path has been removed. A future
checkpoint path should be introduced only after the checkpoint concept is
defined.

## Desired Direction

Signal-discovery-derived strategies should eventually flow through the same
strategy backtest boundary as hand-written strategies.

Signal discovery should produce or select strategy state. It should not remain
the reason a strategy uses a separate evaluation path.

## Close Condition

Close this when alpha-os has a clear boundary between candidate production and
the engine path used to evaluate a fixed candidate.
