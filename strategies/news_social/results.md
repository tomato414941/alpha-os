# News And Social Attention Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.news_social.current_attention_snapshot
uv run python -m strategies.news_social.current_attention_market_join
uv run python -m strategies.news_social.current_attention_forward_labels
```

Interpretation:

- Fear & Greed is a market-level sentiment proxy
- CoinGecko trending is an attention proxy, not an alpha signal by itself
- the next useful test is event-to-return labeling with no lookahead
- paid or authenticated feeds may be needed for serious news/social work

## Snapshot

| source | rank | symbol | name | score | label/value |
| --- | ---: | --- | --- | ---: | --- |
| alternative_me_fear_greed | 1 | MARKET | Crypto Fear and Greed Index | 12 | Extreme Fear |
| coingecko_trending | 1 | PENGU | Pudgy Penguins | 1 | 7.3790 |
| coingecko_trending | 2 | BTC | Bitcoin | 1 | 1.9334 |
| coingecko_trending | 3 | LAB | LAB | 2 | 36.8998 |
| coingecko_trending | 4 | NEAR | NEAR Protocol | 3 | -2.3952 |

This gives the project a non-price attention input. The next step is to label
whether attention spikes lead, lag, or merely coincide with returns and funding
states.

## Attention Market Join

This joins CoinGecko trending assets to current Hyperliquid perp market state.

| symbol | name | rank | 24h change | funding | mark/oracle | carry action | obs | score | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| AERO | Aerodrome Finance | 14 | 3.8259 | -0.288468 | -0.001183 | long_carry_reversion_watch | 6 | 20.605424 | trending asset overlaps with persistent carry/reversion perp state |

Interpretation:

- `AERO` currently overlaps attention and persistent perp carry/reversion state.
- This is not yet a strategy. It needs future-return labels and execution checks.

## Attention Forward Labels

This labels attention/perp-overlap candidates with subsequent Hyperliquid
returns. Positive directional return means the carry or funding direction was
right over that horizon.

| symbol | action | dir | score | raw 15m | dir 15m | raw 1h | dir 1h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AERO | attention_carry_reversion_watch | 1 | 20.605424 | 0.001880 | 0.001880 | -0.001152 | -0.001152 |

Interpretation:

- `AERO` was positive over 15m but negative over 1h.
- This points to a possible short-lived attention/carry overlap, not a stable
  deployable strategy.
