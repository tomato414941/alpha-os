# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 50.4622 | 1 | 7.72 | -21.62 | -29.67 | 0.1549 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 37.0000 | 9 | 61.18 | 167.08 | 354.61 | 2.0431 | avoid chasing ALLO; label pullback or fade setups instead |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 36.1087 | 7 | 13.15 | -7.11 | 35.46 | 0.2318 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| BONK | Bonk | attention_price_lag_candidate | long_attention_lag | 36.0855 | 3 | 1.18 | -18.42 | -39.04 | 0.0698 | paper-label BONK attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 35.3312 | 2 | 1.24 | -10.10 | -36.21 | 0.2857 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 34.5716 | 5 | 67.07 | 266.77 | 719.79 | 0.0714 | avoid chasing BEAT; label pullback or fade setups instead |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 31.7816 | 6 | 4.29 | -14.68 | 40.17 | 0.0562 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 30.3248 | 10 | 1.85 | -18.54 | -29.21 | 0.0787 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 27.9529 | 4 | 1.44 | -13.13 | -21.16 | 0.0276 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
