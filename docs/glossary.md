# Glossary

## trading strategy

A policy-like decision component that consumes observations and optional
strategy state, then produces trading actions and optional next strategy state.

It may include alpha logic, portfolio construction, and execution strategy. It
does not include the executor, fills, market environment, runtime, or evaluation
settings.

In ML/RL terms, a trading strategy is closest to a policy.
