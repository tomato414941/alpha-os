# Liquidation Flow Results

Run:

```bash
uv run python -m strategies.liquidation_flow.current_okx_liquidation_flow
uv run python -m strategies.liquidation_flow.current_okx_liquidation_forward_labels
uv run python -m strategies.liquidation_flow.current_okx_liquidation_monitor
uv run python -m strategies.liquidation_flow.current_okx_liquidation_depth_check
uv run python -m strategies.liquidation_flow.current_okx_liquidation_actionability_review
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
| WLD | short_liquidation_squeeze_watch | 1 | 0.027309 | 0.027309 |  |  |
| ALLO | short_liquidation_squeeze_watch | 1 | 0.019775 | 0.019775 |  |  |
| H | short_liquidation_squeeze_watch | 1 | 0.013053 | 0.013053 |  |  |
| LAB | short_liquidation_squeeze_watch | 1 | 0.004624 | 0.004624 |  |  |
| BEAT | short_liquidation_squeeze_watch | 1 | 0.004041 | 0.004041 |  |  |
| PEPE | short_liquidation_squeeze_watch | 1 | 0.003269 | 0.003269 | 0.005449 | 0.005449 |
| SOL | short_liquidation_squeeze_watch | 1 | 0.001699 | 0.001699 | 0.005406 | 0.005406 |
| ZEC | short_liquidation_squeeze_watch | 1 | -0.006303 | -0.006303 |  |  |
| HOME | long_liquidation_cascade_watch | -1 | 0.007351 | -0.007351 |  |  |
| NEAR | short_liquidation_squeeze_watch | 1 | -0.009277 | -0.009277 |  |  |

Interpretation:

- `WLD`, `ALLO`, `H`, `LAB`, `BEAT`, `PEPE`, and `SOL` had positive first 15m
  continuation labels.
- `ZEC` had one of the largest liquidation-flow scores, but its first 15m
  continuation label is negative.
- The current evidence suggests liquidation flow may be useful, but the best
  continuation rows are not necessarily the largest raw liquidation bursts.

## Current OKX Liquidation Monitor

This repeats the OKX liquidation-flow screen over a short window. It is a
persistence check, not a trade instruction.

| asset | action | obs | mean score | min score | mean liq USD | mean liq/vol | mean imbalance | latest liquidation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BEAT | short_liquidation_squeeze_watch | 3 | 0.127525 | 0.127223 | 174480 | 0.000592 | 1.000000 | 2026-06-07T15:43:11.119000+00:00 |
| WLD | short_liquidation_squeeze_watch | 3 | 0.117841 | 0.115623 | 522179 | 0.001052 | 0.635519 | 2026-06-07T15:46:29.788000+00:00 |
| BSB | mixed_liquidation_flow_watch | 3 | 0.109096 | 0.108608 | 597731 | 0.002554 | 0.373725 | 2026-06-07T15:45:43.900000+00:00 |
| JTO | long_liquidation_cascade_watch | 3 | 0.094599 | 0.094547 | 41203 | 0.000710 | -0.769187 | 2026-06-07T15:44:30.934000+00:00 |
| ZEC | mixed_liquidation_flow_watch | 3 | 0.050522 | 0.050520 | 403706 | 0.000380 | 0.462134 | 2026-06-07T15:45:59.149000+00:00 |
| H | short_liquidation_squeeze_watch | 3 | 0.021281 | 0.021270 | 4007 | 0.000035 | 1.000000 | 2026-06-07T15:44:22.067000+00:00 |
| HOME | mixed_liquidation_flow_watch | 3 | 0.018062 | 0.018056 | 17003 | 0.000305 | -0.244667 | 2026-06-07T15:44:07.932000+00:00 |

Interpretation:

- `BEAT` and `WLD` persisted in all monitor samples as
  `short_liquidation_squeeze_watch`.
- `WLD` is the stronger cross-lane candidate because it also has positive
  pressure and candidate-validation labels.
- `BEAT` is a cleaner liquidation-flow follow-up candidate, but it needs
  depth/fee checks and forward labels from the monitor timestamps.

## Current OKX Liquidation Depth Check

This checks visible OKX book depth for liquidation-monitor candidates. It is
not a fill guarantee.

| asset | action | spread bps | bid depth 5bps | ask depth 5bps | bid depth 10bps | ask depth 10bps | monitor score | depth score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LTC | long_liquidation_cascade_watch | 2.3824 | 33473 | 26234 | 92199 | 95779 | 0.035068 | 0.029440 |
| ONDO | short_liquidation_squeeze_watch | 2.8806 | 14318 | 13968 | 29835 | 26141 | 0.016723 | 0.011611 |
| ZEC | mixed_liquidation_flow_watch | 0.2362 | 16507 | 56008 | 82299 | 135785 | 0.050522 | 0.008747 |
| JTO | long_liquidation_cascade_watch | 1.6149 | 2916 | 2208 | 9749 | 7819 | 0.094599 | 0.003139 |
| H | short_liquidation_squeeze_watch | 1.7710 | 1565 | 767 | 8137 | 4446 | 0.021281 | 0.002300 |
| BEAT | short_liquidation_squeeze_watch | 2.1265 | 846 | 1778 | 9304 | 7679 | 0.127525 | 0.000291 |
| WLD | short_liquidation_squeeze_watch | 2.0728 | 2430 | 2900 | 19616 | 10034 | 0.117841 | 0.000265 |

Interpretation:

- `BEAT` and `WLD` have strong liquidation persistence, but visible 5bps OKX
  depth is small relative to the liquidation burst size.
- `LTC`, `ONDO`, and `ZEC` have better visible-depth profiles, but their signal
  quality is less clean than `WLD`/`BEAT`.
- The immediate paper-trade question is now sizing and venue depth, not only
  whether the signal exists.

## Current OKX Liquidation Actionability Review

This joins liquidation persistence, first continuation labels, and visible
near-touch depth. It is a triage view, not an order plan.

| asset | action | obs | monitor score | cont15 | spread bps | near depth 5bps | score | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LTC | long_liquidation_cascade_watch | 3 | 0.035068 | 0.002599 | 2.3824 | 26234 | 0.926541 | first checks support follow-up |
| WLD | short_liquidation_squeeze_watch | 3 | 0.117841 | 0.027309 | 2.0728 | 2430 | 0.781267 | signal ok but visible depth thin |
| ONDO | short_liquidation_squeeze_watch | 3 | 0.016723 | 0.002006 | 2.8806 | 13968 | 0.541753 | first checks support follow-up |
| H | short_liquidation_squeeze_watch | 3 | 0.021281 | 0.013053 | 1.7710 | 767 | 0.325660 | signal ok but visible depth thin |
| BEAT | short_liquidation_squeeze_watch | 3 | 0.127525 | 0.004041 | 2.1265 | 846 | 0.248129 | signal ok but visible depth thin |
| JTO | long_liquidation_cascade_watch | 3 | 0.094599 | 0.000323 | 1.6149 | 2208 | 0.237773 | signal ok but visible depth thin |
| ZEC | mixed_liquidation_flow_watch | 3 | 0.050522 |  | 0.2362 | 16507 | 0.075261 | waiting for matching forward label |

Interpretation:

- `LTC` and `ONDO` are more executable-looking liquidation follow-ups because
  first labels are positive and visible near-touch depth is stronger.
- `WLD` remains the strongest cross-lane alpha candidate, but OKX near-touch
  depth is thin enough that it should be treated as a small-size or alternate
  venue probe.
- `BEAT` has persistent squeeze flow but weak visible near-touch depth.
