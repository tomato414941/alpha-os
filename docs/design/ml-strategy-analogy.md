# ML Strategy Analogy

Use machine learning vocabulary as a design analogy, not as a new domain model.

## Terms

- strategy: analogous to an inference pipeline. It turns market inputs into a
  portfolio decision.
- strategy checkpoint: analogous to a model checkpoint. It is saved strategy
  state that can be loaded later for backtest, evaluation, or run.

## Boundary

An evaluator should receive a strategy, an optional strategy checkpoint, and
data inputs. It should not need to know whether the checkpoint came from signal
discovery, fixed-state replay, or a hand-written strategy.
