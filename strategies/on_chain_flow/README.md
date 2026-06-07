# On-Chain Flow

This lane looks for capital movement that is not just price momentum.

The current probe uses DeFiLlama chain TVL as a rough capital-flow source. It is
not a trading strategy by itself; it is a candidate source for regime, breadth,
and chain-token follow-up.

## Commands

```bash
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow_venue_coverage
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow_forward_labels
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow_market_context
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow_market_context_history
```
