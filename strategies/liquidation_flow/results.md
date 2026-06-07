# Liquidation Flow Results

Run:

```bash
uv run python -m strategies.liquidation_flow.current_okx_liquidation_flow
uv run python -m strategies.liquidation_flow.current_okx_liquidation_forward_labels
```

This lane looks for recent forced-liquidation bursts. It is not yet a final
alpha test because it has no post-event return labels.

## Current OKX Liquidation Flow

This maps recent OKX USDT swap liquidation flow. Long liquidation means forced
sell flow; short liquidation means forced buy flow.

| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | short_liquidation_squeeze_watch | 440 | 112707 | 1193458 | 1306165 | 0.001213 | 0.827424 | 0.176258 |
| WLD | short_liquidation_squeeze_watch | 257 | 65082 | 432606 | 497689 | 0.001053 | 0.738462 | 0.136540 |
| JTO | long_liquidation_cascade_watch | 69 | 49065 | 4117 | 53183 | 0.000935 | -0.845163 | 0.122133 |
| BEAT | short_liquidation_squeeze_watch | 96 | 0 | 135450 | 135450 | 0.000474 | 1.000000 | 0.111679 |
| BSB | mixed_liquidation_flow_watch | 504 | 152970 | 295347 | 448316 | 0.002350 | 0.317581 | 0.087000 |
| HOME | long_liquidation_cascade_watch | 80 | 32025 | 6585 | 38610 | 0.000674 | -0.658901 | 0.078440 |
| ONDO | short_liquidation_squeeze_watch | 2 | 0 | 10345 | 10345 | 0.000273 | 1.000000 | 0.066295 |
| OPN | long_liquidation_cascade_watch | 39 | 17755 | 1283 | 19038 | 0.000288 | -0.865203 | 0.062830 |
| BTC | short_liquidation_squeeze_watch | 160 | 263236 | 927918 | 1191155 | 0.000186 | 0.558015 | 0.046283 |

Interpretation:

- `ZEC` and `WLD` have the strongest recent liquidation-flow scores.
- `WLD` also appears in candidate validation and OKX perp pressure, so it is now
  a cross-lane follow-up candidate.
- This still does not decide continuation vs reversal. The next step is
  post-liquidation return labels joined to funding, open interest, and book
  depth.

## Current OKX Liquidation Forward Labels

This labels liquidation-flow candidates with continuation returns. Positive
continuation return means the forced-flow direction continued over that horizon.

| asset | action | dir | raw 15m | continuation 15m | raw 1h | continuation 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ALLO | short_liquidation_squeeze_watch | 1 | 0.019775 | 0.019775 |  |  |
| H | short_liquidation_squeeze_watch | 1 | 0.013053 | 0.013053 |  |  |
| BEAT | short_liquidation_squeeze_watch | 1 | 0.006535 | 0.006535 |  |  |
| EDEN | long_liquidation_cascade_watch | -1 | -0.005715 | 0.005715 |  |  |
| PEPE | short_liquidation_squeeze_watch | 1 | 0.003269 | 0.003269 | 0.003269 | 0.003269 |
| SOL | short_liquidation_squeeze_watch | 1 | 0.001699 | 0.001699 | 0.003707 | 0.003707 |
| ZEC | short_liquidation_squeeze_watch | 1 | -0.000070 | -0.000070 |  |  |
| WLD | short_liquidation_squeeze_watch | 1 | -0.001606 | -0.001606 |  |  |
| HOME | long_liquidation_cascade_watch | -1 | 0.003196 | -0.003196 |  |  |
| JTO | long_liquidation_cascade_watch | -1 | 0.003874 | -0.003874 |  |  |

Interpretation:

- `ALLO`, `H`, `BEAT`, `EDEN`, `PEPE`, and `SOL` had positive first 15m
  continuation labels.
- `ZEC` and `WLD` had the largest liquidation-flow scores, but their first 15m
  continuation labels were slightly negative.
- The current evidence suggests liquidation flow may be useful, but the best
  continuation rows are not necessarily the largest raw liquidation bursts.
