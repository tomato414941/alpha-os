# Strategies

This directory is for concrete strategy implementations that use alpha-os.

It is not package API and not a usage example directory:

- library contracts belong in `src/alpha_os/`
- usage sketches belong in `examples/`
- frozen historical research belongs in `experiments/`
- profit-seeking strategy candidates belong here

A maintained strategy should eventually include:

- a short hypothesis
- the market data it expects
- a concrete `TradingStrategy`-style boundary when that fits
- a local backtest or smoke path
- result notes for iteration decisions

Do not move shared code into `src/alpha_os/` just because one strategy uses it.
Promote code to the library only after multiple strategies need the same shape.

## Current Candidates

- `crypto/`
  - crypto long-or-cash momentum and allocation variants
- `crypto_pair_spread/`
  - crypto relative-value spread candidate
- `equity_index/`
  - ETF momentum rotation candidate
- `cash_rotation/`
  - risk-on/risk-off rotation candidate
- `cross_asset_rotation/`
  - mixed ETF and crypto daily close rotation candidates
- `crypto_market_structure/`
  - crypto futures funding, premium, taker-flow, and volume diagnostics
- `research_map.md`
  - broad research lanes, profit-source priorities, and anti-constraints
- `leaderboard.py`
  - cross-strategy comparison against same-window buy-and-hold benchmarks
- `leaderboard.md`
  - latest broad comparison result

## Local Shared Strategy Code

- `daily_close/`
  - daily close data, backtest, metrics, and Yahoo fetch utilities used by
    concrete strategies
  - this is strategy implementation support, not `alpha_os` package API
