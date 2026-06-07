# Perp Market Map Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.perp_market_map.current_hyperliquid_snapshot
```

Interpretation:

- positive funding means short perp receives funding
- negative funding means long perp receives funding
- high open interest and volume improve feasibility
- wide impact spread weakens feasibility
- this is a current snapshot, not a historical backtest

## Top Snapshot Rows

| asset | annualized funding | 24h notional volume | impact spread | carry side |
| --- | ---: | ---: | ---: | --- |
| MANTA | 3.2319 | 652166 | 0.003869 | short perp receives funding |
| STABLE | -2.8106 | 1321710 | 0.002687 | long perp receives funding |
| BSV | -1.8717 | 211091 | 0.003335 | long perp receives funding |
| ZORA | -1.6818 | 270157 | 0.003177 | long perp receives funding |
| MEME | -1.2334 | 580003 | 0.004894 | long perp receives funding |

The immediate next question is not whether these rows are profitable today. The
right question is whether large funding, premium, open interest, and impact
spread states persist long enough to execute and hedge.

