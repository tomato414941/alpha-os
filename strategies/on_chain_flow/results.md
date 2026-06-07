# On-Chain Flow Results

Run:

```bash
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow
```

This is not a deployable strategy. It searches for broad chain-level capital
movement that can become a candidate source when joined to token returns,
funding, liquidation, and breadth.

## Current Chain TVL Flow

| chain | token | tvl USD | day % | week % | month % | action | followup |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ENI | ENI | 320783901 | 0.0302 | 0.2403 | 1.5101 | chain_inflow_momentum_watch | label ENI continuation against perp funding and liquidity |
| X Layer |  | 72267519 | -0.0459 | -0.2063 | -0.0804 | chain_outflow_stress_watch | use as broad on-chain regime context |
| Avalanche | AVAX | 478121689 | -0.0126 | -0.2054 | -0.2532 | chain_outflow_stress_watch | label AVAX downside or rotation-away behavior |
| Hydration | HDX | 54901126 | -0.0007 | -0.1859 | -0.2670 | chain_outflow_stress_watch | label HDX downside or rotation-away behavior |
| Rootstock | RBTC | 93639147 | -0.0019 | -0.1392 | -0.2219 | chain_outflow_stress_watch | label RBTC downside or rotation-away behavior |
| Near | NEAR | 143497256 | -0.0212 | -0.1317 | 0.0429 | chain_outflow_stress_watch | label NEAR downside or rotation-away behavior |
| Tron | TRON | 4357895797 | -0.0002 | -0.1029 | -0.1560 | chain_outflow_stress_watch | label TRON downside or rotation-away behavior |
| Stellar | XLM | 201515549 | -0.0040 | -0.0772 | 0.0853 | chain_outflow_stress_watch | label XLM downside or rotation-away behavior |
| Hyperliquid L1 | HYPE | 1516403481 | 0.0129 | -0.1097 | -0.0095 | chain_flow_reversal_watch | separate HYPE reversal from stale TVL accounting |

Interpretation:

- `ENI` is the only top-chain clear inflow momentum row in this run.
- `AVAX`, `NEAR`, `TRON`, and `XLM` show weekly TVL outflow with current stress
  direction.
- `HYPE`, `SOL`, `ETH`, `BNB`, and other large chains show weekly outflow but
  daily rebound. That is a reversal/divergence setup, not a simple short.
- `XLM` is especially important because candidate repeat labels were positive
  while chain TVL flow is negative. That conflict should be isolated before
  promoting XLM.
