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

The long-run allocation unit should be `subject_id`, not `asset`.

`asset` is an acceptable convenience field for the current bounded single-asset
runtime, but it should not be treated as the long-run source-of-truth concept.
The durable concept is the subject that can carry portfolio weight.

Examples of valid subjects include:

- a spot asset such as `BTC`
- another asset such as `GLD` or a REIT ETF
- an instrument such as `BTC_perp`
- a basket or sleeve such as `REIT_basket`
- a pair or spread such as `ETH_BTC_pair`

So in a broader runtime:

- `portfolio_id` identifies which capital pool is being managed
- `subject_id` identifies what can receive weight
- `asset` becomes, at most, a convenience field for bounded cases

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

## Initial Problem Statement

The first explicit portfolio problem should be modeled as a constrained
single-period portfolio choice with state carry-over.

That means:

- the system chooses a desired portfolio for the next decision step
- the current portfolio remains an input
- the optimizer is not yet a full multi-period control system
- turnover and no-trade logic still matter because rebalancing is stateful

This is a better starting point than either:

- a pure stateless mapping from prediction to trade
- a fully general dynamic programming problem

## Initial Objective

The initial objective should be:

- maximize expected portfolio reward
- penalize ex-ante portfolio risk
- penalize turnover and execution cost
- penalize low-confidence exposures

In words:

- prefer portfolios with better expected edge
- prefer portfolios that achieve that edge with lower concentration and lower
  risk
- avoid paying cost for weak or uncertain changes

This should be treated as a risk-adjusted utility problem, not as raw return
maximization.

## Initial Risk Model

The first risk model should be simple but explicit.

It should include:

- ex-ante portfolio variance or another covariance-based dispersion term
- position concentration penalties
- optional volatility scaling at the asset or sleeve level

The first version does not need a full tail-risk model, but it should leave
room for later drawdown or downside-aware extensions.

## Initial Dependence Model

The portfolio layer should assume that signals and positions are not
independent.

The first dependence model should therefore allow:

- cross-asset covariance
- cross-target correlation where relevant
- signal crowding or overlap penalties when multiple predictive objects imply
  the same hidden bet

The point is not to eliminate similarity. The point is to avoid counting many
similar signals as independent edge.

## Initial Cost Model

The first cost model should be simple and explicit.

It should include:

- turnover penalty
- proportional slippage estimate
- no-trade region for small desired changes

The portfolio layer should decide that some nominal improvements are not worth
trading once cost is considered.

## Initial Uncertainty Model

The first uncertainty model should discount weak evidence before it reaches
execution.

It should allow:

- lower influence for targets or ensembles with unstable scoring
- lower influence for hypotheses with small or fragile samples
- confidence-aware shrinkage rather than binary trust in point estimates

This is the main reason predictive quality should not map directly to position
size.

## Initial Constraints

The first version should explicitly support constraints such as:

- gross exposure bounds
- net exposure bounds
- maximum single-position weight
- turnover cap
- liquidity or tradeability cap

The exact numbers can remain open. The important point is that these are part
of the portfolio problem, not execution afterthoughts.

## Initial Time Model

The first portfolio decision layer should be rolling and state-aware.

That means:

- each decision is made at a point in time
- the current portfolio is an input
- the next desired portfolio is an output
- costs depend on the path between them

This is enough to capture rebalancing logic without prematurely committing to a
fully dynamic optimal control formulation.

## Current Runtime Coverage

The current runtime already provides some inputs that the portfolio layer will
eventually need.

### Already Present

- target definitions
- hypothesis predictions
- target-level meta predictions
- target-level scoring for hypotheses
- target-level scoring for meta aggregations
- rolling history of realized outcomes

This means the project already has a usable predictive substrate.

### Missing Or Weak

The portfolio layer still lacks several inputs that are required by the initial
problem statement.

- **current portfolio state**
  - there is no first-class portfolio state object yet
- **risk model inputs**
  - there is no covariance, volatility, or concentration input for decision use
- **cost model inputs**
  - there is no explicit turnover or slippage model in the decision layer
- **uncertainty inputs**
  - scoring exists, but confidence-aware shrinkage is not defined
- **dependence inputs**
  - hidden-bet overlap is not represented as a portfolio input
- **decision output**
  - there is no first-class desired portfolio state such as target weights or
    position deltas

So the current runtime is strong on prediction and evaluation, but weak on
portfolio-state inputs and decision outputs.

## Next Design Step

The next design step should not be to force decisions directly from the current
runtime.

Instead, it should define the missing contract between:

- predictive layer outputs
- current portfolio state
- risk and cost inputs
- desired portfolio state outputs

The smallest useful next artifact is therefore:

- a portfolio state object
- a portfolio decision input object
- a portfolio decision output object

Only after that contract exists should the first decision rule be implemented.
