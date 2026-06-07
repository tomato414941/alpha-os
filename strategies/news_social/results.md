# News And Social Attention Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.news_social.current_attention_snapshot
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

