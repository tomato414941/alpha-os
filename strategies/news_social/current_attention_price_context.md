# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 47.3497 | 2 | 6.10 | -22.25 | -29.97 | 0.1451 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 40.3113 | 5 | 11.43 | -9.31 | 34.45 | 0.2346 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 37.1249 | 1 | 1.33 | -10.80 | -35.54 | 0.2809 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 36.5528 | 3 | 72.80 | 278.46 | 747.51 | 0.0711 | avoid chasing BEAT; label pullback or fade setups instead |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 36.0000 | 10 | 73.62 | 191.67 | 397.72 | 2.6977 | avoid chasing ALLO; label pullback or fade setups instead |
| BONK | Bonk | attention_price_lag_candidate | long_attention_lag | 33.2575 | 7 | 1.47 | -19.24 | -38.56 | 0.0710 | paper-label BONK attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 32.9959 | 6 | 4.12 | -16.07 | 40.30 | 0.0560 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 31.5665 | 9 | 2.23 | -18.40 | -29.09 | 0.0788 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 27.7393 | 4 | 1.16 | -13.22 | -21.25 | 0.0272 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
