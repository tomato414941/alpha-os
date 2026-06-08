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
- `basis_term_structure/`
  - Deribit dated-futures basis screen for carry and relative-value candidates
- `cross_exchange_funding/`
  - current predicted funding spread screens across venues
- `perp_market_map/`
  - Hyperliquid perp funding, open interest, volume, premium, and impact-spread
    snapshot
- `derivatives_positioning/`
  - CoinGecko multi-venue derivatives OI, volume, funding, basis, and spread
    screen
- `market_making/`
  - Hyperliquid L2 order-book snapshot and near-book depth diagnostics
- `options_volatility/`
  - Deribit BTC/ETH option IV, skew, and term-structure surface diagnostics
- `sector_rotation/`
  - CoinGecko crypto category rotation snapshot for broad thematic flow context
- `event_flow/`
  - Binance USD-M futures taker-flow data path and first 5-minute imbalance
    diagnostic
- `liquidation_flow/`
  - OKX forced-liquidation flow screen for cascade and squeeze candidates
- `defi_yield/`
  - DeFiLlama stable-yield pool screen and peg-risk join
- `defi_lending/`
  - Morpho lending and borrowing rate pressure screen
- `dex_pool_flow/`
  - GeckoTerminal DEX pool liquidity, volume, turnover, and price-flow screen
- `protocol_fundamentals/`
  - DeFiLlama protocol fee-growth screen mapped to tradable token candidates
- `news_social/`
  - crypto RSS news-event, Fear & Greed, and CoinGecko trending attention
    screens
- `market_breadth/`
  - broad CoinGecko volume-price dislocation screen for reversal,
    continuation, and chase-risk candidates
- `anomaly_stress/`
  - cross-market anomaly screen across peg, lending, yield, volatility,
    prediction-market, and execution-spread probes
- `stablecoin_liquidity/`
  - DeFiLlama stablecoin supply, peg, supply-change, and chain migration
    snapshots
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
- `current_alpha_stack.py`
  - cross-lane current paper candidate stack generator
- `current_alpha_stack.md`
  - latest cross-lane paper candidate stack
- `current_paper_probe_plan.py`
  - cross-lane queue for current small paper observations
- `current_paper_probe_plan.md`
  - latest current paper-observation plan
- `current_paper_tickets.py`
  - opens the current paper-observation plan into timestamped paper tickets
- `current_paper_tickets.md`
  - latest current paper tickets with entry marks and required records
- `current_paper_ticket_outcomes.py`
  - checks opened paper tickets against latest public marks after checkpoints
- `current_paper_ticket_outcomes.md`
  - latest paper-ticket mark outcomes and remaining evidence gaps
- `current_paper_ticket_action_queue.py`
  - turns paper-ticket outcomes into promotion, repeat, deprioritization, or wait actions
- `current_paper_ticket_action_queue.md`
  - latest action queue from paper-ticket outcomes
- `current_observation_cycle.py`
  - refreshes current stack, plan, ticket outcomes, symbol queues, and board
    without reopening paper tickets unless `--open-new-tickets` is passed;
    use `--refresh-public-marks` before checkpoint outcome checks
- `current_symbol_opportunity_map.py`
  - symbol-level cluster generator from the current alpha stack
- `current_symbol_opportunity_map.md`
  - latest symbol-level cluster map for repeated labels and execution checks
- `current_symbol_cluster_conflicts.py`
  - symbol-level confirmation and conflict screen from the current alpha stack
- `current_symbol_cluster_conflicts.md`
  - latest symbol-level direction and structure conflict screen
- `current_symbol_cluster_label_queue.py`
  - symbol-level label-work queue from the current conflict screen
- `current_symbol_cluster_label_queue.md`
  - latest symbol-level split-label and confirmed-direction label queue
- `current_symbol_lane_split_review.py`
  - lane-level review for the top symbol-label queue
- `current_symbol_lane_split_review.md`
  - latest lane-level split of conflicting symbol alpha ideas
- `p0_parallel/`
  - parallel P0 data reachability, L2 burst, and operational paper-ticket
    probes

## Local Shared Strategy Code

- `daily_close/`
  - daily close data, backtest, metrics, and Yahoo fetch utilities used by
    concrete strategies
  - this is strategy implementation support, not `alpha_os` package API
