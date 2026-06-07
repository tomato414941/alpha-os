# On-Chain Flow Results

Run:

```bash
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow_venue_coverage
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow_forward_labels
uv run python -m strategies.on_chain_flow.current_chain_tvl_flow_market_context
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

## Venue Coverage

| chain | token | action | week % | day % | HL | OKX | venues | followup |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| Avalanche | AVAX | chain_outflow_stress_watch | -0.2054 | -0.0126 | True | True | 2 | label AVAX short stress on covered venues |
| Near | NEAR | chain_outflow_stress_watch | -0.1317 | -0.0212 | True | True | 2 | label NEAR short stress on covered venues |
| Stellar | XLM | chain_outflow_stress_watch | -0.0772 | -0.0040 | True | True | 2 | label XLM short stress on covered venues |
| Starknet | STRK | chain_outflow_stress_watch | -0.0621 | -0.0032 | True | True | 2 | label STRK short stress on covered venues |
| Cardano | ADA | chain_flow_reversal_watch | -0.2861 | 0.0065 | True | True | 2 | label ADA rebound continuation on covered venues |
| MegaETH | MEGA | chain_flow_reversal_watch | -0.2000 | 0.0019 | True | True | 2 | label MEGA rebound continuation on covered venues |
| Hyperliquid L1 | HYPE | chain_flow_reversal_watch | -0.1097 | 0.0129 | True | True | 2 | label HYPE rebound continuation on covered venues |
| ENI | ENI | chain_inflow_momentum_watch | 0.2403 | 0.0302 | False | False | 0 | keep as context until a perp venue exists |

Interpretation:

- `ENI` is the strongest TVL inflow row, but it is not currently covered by HL
  or OKX USDT perps in this check.
- `AVAX`, `NEAR`, `XLM`, and `STRK` are the tradable outflow-stress candidates.
- `ADA`, `MEGA`, and `HYPE` are tradable weekly-outflow/daily-rebound reversal
  candidates.
- This screen still needs token forward labels and cost checks before any
  promotion.

## Forward Labels

| venue | chain | token | action | dir | week % | day % | raw 15m | dir 15m | status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| HL | Hyperliquid L1 | HYPE | chain_flow_reversal_watch | 1 | -0.1097 | 0.0129 | 0.004285 | 0.004285 | labeled_15m_pending_1h |
| OKX | Hyperliquid L1 | HYPE | chain_flow_reversal_watch | 1 | -0.1097 | 0.0129 | 0.004070 | 0.004070 | labeled_15m_pending_1h |
| OKX | Katana | KAT | chain_flow_reversal_watch | 1 | -0.3122 | 0.0369 | 0.003103 | 0.003103 | labeled_15m_pending_1h |
| OKX | Polygon | POL | chain_flow_reversal_watch | 1 | -0.0754 | 0.0157 | 0.002530 | 0.002530 | labeled_15m_pending_1h |
| HL | OP Mainnet | OP | chain_flow_reversal_watch | 1 | -0.1227 | 0.0005 | 0.002521 | 0.002521 | labeled_15m_pending_1h |
| HL | MegaETH | MEGA | chain_flow_reversal_watch | 1 | -0.2000 | 0.0019 | 0.002481 | 0.002481 | labeled_15m_pending_1h |
| HL | Starknet | STRK | chain_outflow_stress_watch | -1 | -0.0621 | -0.0032 | -0.001801 | 0.001801 | labeled_15m_pending_1h |
| OKX | Avalanche | AVAX | chain_outflow_stress_watch | -1 | -0.2054 | -0.0126 | 0.001946 | -0.001946 | labeled_15m_pending_1h |
| OKX | Near | NEAR | chain_outflow_stress_watch | -1 | -0.1317 | -0.0212 | 0.003929 | -0.003929 | labeled_15m_pending_1h |

Interpretation:

- The first 15m labels favor `chain_flow_reversal_watch`, not
  `chain_outflow_stress_watch`.
- `HYPE`, `KAT`, `POL`, `OP`, and `MEGA` are the current tradable winners from
  the TVL-flow reversal family.
- `STRK` is the only outflow-stress short with a positive 15m directional label
  in this refresh; `AVAX`, `NEAR`, and `XLM` failed.
- This strengthens the broader divergence thesis: weekly TVL outflow plus daily
  rebound may be more useful than raw outflow-as-short.

## Market Context

| venue | token | action | dir15 | funding support | funding | liq action | liq score | score | note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| OKX | SOL | chain_flow_reversal_watch | 0.00107280 | 0.38634890 | -0.38634890 |  |  | 0.417124 | price label positive; funding helps direction |
| OKX | HYPE | chain_flow_reversal_watch | 0.00406987 | 0.06081031 | -0.06081031 | mixed_liquidation_flow_watch | 0.00082611 | 0.391720 | price label positive; funding helps direction |
| OKX | ETH | chain_flow_reversal_watch | 0.00032510 | 0.15622280 | -0.15622280 | short_liquidation_squeeze_watch | 0.01962368 | 0.381906 | price label positive; funding helps direction; has recent liquidation context |
| OKX | MEGA | chain_flow_reversal_watch | 0.00206697 | 0.26241290 | -0.26241290 |  |  | 0.366388 | price label positive; funding helps direction |
| HL | HYPE | chain_flow_reversal_watch | 0.00428494 | -0.10950000 | 0.10950000 | mixed_liquidation_flow_watch | 0.00082611 | 0.318829 | price label positive |

Interpretation:

- `SOL`, `HYPE`, `ETH`, and `MEGA` currently combine positive chain-flow
  reversal labels with funding that helps a long direction on OKX.
- `ETH` has the clearest recent liquidation context among the top rows.
- `KAT` and `POL` still have positive price labels, but their OKX pressure
  context is missing from the current top-volume pressure snapshot.
