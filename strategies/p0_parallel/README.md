# P0 Parallel

This directory runs the P0 lanes from `candidate_matrix.md` in parallel.

It is deliberately not a single-strategy folder. The point is to avoid another
narrow deep dive before data reachability, execution feasibility, and first
falsification tests are visible across the highest-priority lanes.

## Commands

```bash
uv run python -m strategies.p0_parallel.data_reachability_probe
uv run python -m strategies.p0_parallel.binance_derivatives_history_probe
uv run python -m strategies.p0_parallel.l2_burst_probe
uv run python -m strategies.p0_parallel.paper_trade_ticket
```

## Lanes Covered

- liquidation/OI/funding data reachability
- multi-venue funding/basis data reachability
- Binance USD-M metrics, premium-index, and funding-rate history first label
- L2 fill/adverse-selection first burst
- paper/manual trade-ticket feasibility
- attention/liquidity inputs through existing stablecoin and sentiment probes
