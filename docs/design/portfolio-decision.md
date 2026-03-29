# Portfolio Decision

## Purpose

The portfolio decision layer maps predictive objects into desired portfolio
state.

Its job is not to produce raw orders. Its job is to choose what portfolio
should exist now, given the current predictive view and portfolio constraints.

So the intended separation is:

- `hypothesis`
  - produces predictions
- `meta prediction`
  - aggregates predictive views
- `portfolio decision`
  - chooses desired portfolio state
- `execution`
  - converts desired state into bounded orders

## What It Is

Portfolio decision is the layer that determines outputs such as:

- target weights
- target position deltas
- entry or no-trade gates
- risk scaling factors

These are portfolio intents rather than execution instructions.

## What It Is Not

Portfolio decision is not:

- a hypothesis
- a target definition
- a scoring rule
- a meta prediction
- an execution engine

Those layers answer different questions:

- hypotheses answer how predictions are produced
- targets answer what truth is being predicted
- scoring answers how predictions are judged
- meta prediction answers what the ensemble predicts
- execution answers how to place and manage orders

Portfolio decision answers this question:

- given the predictive view and the portfolio state, what portfolio should be
  held now

## Inputs

The portfolio decision layer should be able to consume a structured set of
portfolio-relevant inputs.

The minimal input model should allow:

- expected return style signals
- risk or volatility signals
- confidence or uncertainty signals
- diversification or dependence signals
- execution quality or cost signals
- current portfolio state

These inputs do not need to come from one target family. Different targets may
feed different parts of the decision.

Examples:

- residual return targets may affect direction and expected reward
- volatility targets may affect size and leverage
- directional targets may act as entry filters
- execution targets may defer or suppress trades

## Outputs

The portfolio decision layer should output a desired portfolio state rather than
raw market actions.

The minimal output contract should support:

- `target_weight`
  - desired portfolio weight for a subject
- `position_delta`
  - change required from the current portfolio state
- `entry_allowed`
  - whether a new exposure is permitted
- `risk_scale`
  - multiplicative risk adjustment applied to downstream sizing

The exact runtime may expose one or more of these, but this is the intended
design surface.

## Theory-Driven Requirements

Portfolio decision should be defined from portfolio theory requirements first,
not from the current runtime surface.

At minimum, the layer should make these questions explicit:

- `objective`
  - what is being optimized
- `risk_model`
  - what notion of risk is controlled
- `dependence_model`
  - how correlation or covariance is represented
- `cost_model`
  - how turnover, slippage, and impact are penalized
- `uncertainty_model`
  - how unstable or weak signals are discounted
- `constraints`
  - leverage, concentration, liquidity, and turnover bounds
- `time_model`
  - whether the decision is one-shot, rolling, or stateful

The runtime should adapt to these requirements. The requirements should not be
chosen to fit whatever the runtime already happens to expose.

## Decision Criteria

The portfolio decision layer should be able to express tradeoffs such as:

- higher expected return versus higher risk
- concentrated conviction versus diversification
- faster adaptation versus lower turnover
- theoretical edge versus executable edge

This is why predictive quality alone is insufficient. A good hypothesis or a
good meta prediction does not automatically imply a good portfolio state.

## Minimal Design Goal

The first practical implementation does not need to solve the full portfolio
problem. But it should preserve the right shape.

The minimal acceptable direction is:

- use predictive objects as inputs
- produce desired portfolio state as outputs
- keep execution separate
- leave room for explicit risk, cost, and uncertainty models

This keeps the system aligned with the correct theory-driven interface even
before the full optimization problem is implemented.
