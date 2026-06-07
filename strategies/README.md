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
- `cross_exchange_funding/`
  - current predicted funding spread screens across venues
- `perp_market_map/`
  - Hyperliquid perp funding, open interest, volume, premium, and impact-spread
    snapshot
- `market_making/`
  - Hyperliquid L2 order-book snapshot and near-book depth diagnostics
- `options_volatility/`
  - Deribit BTC/ETH option IV, skew, and term-structure surface diagnostics
- `event_flow/`
  - Binance USD-M futures taker-flow data path and first 5-minute imbalance
    diagnostic
- `liquidation_flow/`
  - OKX forced-liquidation flow screen for cascade and squeeze candidates
- `defi_yield/`
  - DeFiLlama stable-yield pool screen
- `news_social/`
  - Fear & Greed and CoinGecko trending attention snapshot
- `stablecoin_liquidity/`
  - DeFiLlama stablecoin supply, peg, and supply-change snapshot
- `candidate_validation/`
  - current candidate aggregation and Hyperliquid return/volume context
- `research_map.md`
  - broad research lanes, profit-source priorities, and anti-constraints
- `missing_work_inventory.md`
  - broad inventory of missing profit sources, data, evaluation, operations,
    and external tools
- `candidate_matrix.md`
  - prioritized broad candidate matrix with data needs, execution venues,
    failure modes, and first falsification tests
- `opportunity_map.md`
  - profit-source candidates and reachable public data routes
- `data_source_probe.py`
  - public data route probe for event flow, DeFi, exchange, and perp DEX data
- `data_source_probe.csv`
  - latest probe result
- `leaderboard.py`
  - cross-strategy comparison against same-window buy-and-hold benchmarks
- `leaderboard.md`
  - latest broad comparison result
- `exploration_board.py`
  - broad lane status board generator
- `exploration_board.md`
  - latest lane status board
- `p0_parallel/`
  - parallel P0 data reachability, L2 burst, and operational paper-ticket
    probes

## Local Shared Strategy Code

- `daily_close/`
  - daily close data, backtest, metrics, and Yahoo fetch utilities used by
    concrete strategies
  - this is strategy implementation support, not `alpha_os` package API
