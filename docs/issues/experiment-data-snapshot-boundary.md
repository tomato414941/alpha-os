# Experiment Data Boundary

## Problem

Investment hypotheses need fixed evidence data, but `data/` contains local
runtime state, caches, logs, and old outputs. It is not a reliable source of
truth for research evidence.

## Risk

If an experiment silently uses local DBs or external data, the same hypothesis
may not be reproducible later.

## Guard

Before judging a hypothesis, identify the dataset or data retrieval procedure
used for that hypothesis.

Do not treat `data/` as the source of truth for experiment evidence.

## Next Decision

For `crypto_regime_momentum`, decide whether
`experiments/datasets/ds_crypto_btc_eth_daily_2024_2025/` is sufficient after
the hypothesis is narrowed, or whether a reproducible retrieval procedure is
needed.

## Close Condition

Close this when `crypto_regime_momentum` points to exactly one evidence data
source: either a committed dataset or a reproducible retrieval procedure.

## Later

For the first crypto hypothesis, decide whether to keep a committed dataset
under `experiments/datasets/` or use only a retrieval procedure.
