# P0 Parallel

This directory runs the P0 lanes from `candidate_matrix.md` in parallel.

It is deliberately not a single-strategy folder. The point is to avoid another
narrow deep dive before data reachability, execution feasibility, and first
falsification tests are visible across the highest-priority lanes.

## Commands

```bash
uv run python -m strategies.p0_parallel.data_reachability_probe
uv run python -m strategies.p0_parallel.binance_derivatives_history_probe
uv run python -m strategies.p0_parallel.binance_derivatives_symbol_feature_candidates
uv run python -m strategies.p0_parallel.binance_derivatives_feature_regime_compare
uv run python -m strategies.p0_parallel.binance_derivatives_intraday_feature_labels
uv run python -m strategies.p0_parallel.binance_derivatives_intraday_repeat_compare
uv run python -m strategies.p0_parallel.binance_derivatives_intraday_paper_labels
uv run python -m strategies.p0_parallel.funding_carry_proxy
uv run python -m strategies.p0_parallel.l2_burst_probe
uv run python -m strategies.p0_parallel.paper_trade_ticket
```

## Lanes Covered

- liquidation/OI/funding data reachability
- multi-venue funding/basis data reachability
- Binance USD-M metrics, premium-index, and funding-rate history first label
- Binance USD-M symbol-feature candidate queue for recent-window reruns
- historical-vs-recent derivatives feature regime comparison
- Binance USD-M 5m derivatives features against next-1h labels
- non-overlapping Binance USD-M intraday feature repeat comparison
- cost-aware Binance USD-M intraday feature paper labels
- Binance funding carry proxy with premium-change and rough cost
- L2 fill/adverse-selection first burst
- paper/manual trade-ticket feasibility
- attention/liquidity inputs through existing stablecoin and sentiment probes
