# News And Social Attention Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.news_social.current_attention_snapshot
uv run python -m strategies.news_social.current_attention_market_join
uv run python -m strategies.news_social.current_attention_forward_labels
uv run python -m strategies.news_social.current_exchange_catalyst_snapshot
uv run python -m strategies.news_social.current_exchange_catalyst_market_join
uv run python -m strategies.news_social.current_exchange_catalyst_forward_labels
```

Interpretation:

- Fear & Greed is a market-level sentiment proxy
- CoinGecko trending is an attention proxy, not an alpha signal by itself
- Binance and OKX announcements add exchange-catalyst inputs that are not
  derived from prices
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

## Exchange Catalyst Snapshot

This extracts recent Binance and OKX exchange-announcement catalysts such as
perp launches, spot/listing support, removals, airdrops, campaigns, and network
events.

Interpretation:

- This is a broader external-event lane than CoinGecko trending.
- The raw Binance announcement feed includes some exchange product listings that
  are not useful unless they overlap with a tradable venue.
- The useful filter is the market join: keep only symbols that currently exist
  on Hyperliquid or another executable venue.

## Exchange Catalyst Market Join

This joins exchange-announcement catalysts to current Hyperliquid perp state.

Current useful rows:

- `CHIP`: Binance perp/listing catalyst overlaps a Hyperliquid market.
- `MEGA`: Binance spot/support catalyst overlaps a Hyperliquid market.
- `AI`: Binance and OKX listing/removal catalysts overlap a Hyperliquid market,
  but current labels are still pending.
- `SOL`: Binance removal catalyst is tradable and has a short direction hint.
- `NEAR` and `POL`: network-event catalysts are tradable but lower score.

Interpretation:

- Exchange-catalyst candidates are event-reactive. They should not be treated as
  normal momentum or mean-reversion signals.
- The immediate opportunity is to test whether listing/removal/network events
  create short-lived reactions that survive fees and latency.

## Exchange Catalyst Forward Labels

Current labeled reactions:

| symbol | catalyst | dir 15m | dir 1h | read |
| --- | --- | ---: | ---: | --- |
| MEGA | spot_listing_watch | 0.069414 | -0.037918 | strong 15m pop, then reversal |
| NEAR | network_event_watch | 0.024957 | 0.038601 | positive over both horizons |
| CHIP | spot_listing_watch | 0.006453 | 0.012107 | small positive continuation |
| SOL | exchange_removal_watch | 0.001334 | 0.003052 | small short-direction win |
| POL | network_event_watch | -0.000846 | -0.002960 | failed direction |

Interpretation:

- `MEGA` is the strongest event reaction, but the 1h reversal means it is more
  likely a fast event-reaction candidate than a hold candidate.
- `NEAR` is the cleanest current network-event label because both 15m and 1h
  are positive.
- This lane needs repeated events, venue depth, fee/slippage assumptions, and
  latency-aware execution checks before paper trading.
