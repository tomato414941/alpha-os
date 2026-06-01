# Glossary

## trading strategy

A policy-like decision component that consumes observations and optional
strategy state, then produces trading actions and optional next strategy state.

It may include alpha logic, portfolio construction, and execution strategy. It
does not include the executor, fills, market environment, runtime, or evaluation
settings.

In ML/RL terms, a trading strategy is closest to a policy.

## strategy construction

The process of producing an executable trading strategy.

It may include choosing rules or models, fitting state, selecting signals,
constructing portfolio logic, or creating checkpoints.

Strategy construction is not an evaluation concern. Evaluation should consume
the resulting trading strategy and measure its behavior.

Do not treat strategy construction as a required fixed framework. It may be a
plain function, a script, a factory, a training job, or hand-written code,
depending on the strategy.
