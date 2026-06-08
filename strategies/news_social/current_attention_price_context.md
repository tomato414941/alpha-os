# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 53.1177 | 2 | 18.31 | -23.14 | -28.53 | 0.1824 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 47.1727 | 1 | 10.47 | -12.17 | -33.47 | 0.3595 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 41.1694 | 5 | 7.07 | -18.84 | -27.39 | 0.0852 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 38.9488 | 7 | 10.44 | -9.95 | 30.52 | 0.2458 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| PEPE | Pepe | attention_price_lag_candidate | long_attention_lag | 37.9157 | 10 | 5.26 | -17.51 | -34.41 | 0.1829 | paper-label PEPE attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 37.3062 | 4 | 7.65 | -15.02 | 42.34 | 0.0526 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 33.0597 | 3 | 4.85 | -13.71 | -20.50 | 0.0301 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |
| LINK | Chainlink | attention_price_lag_candidate | long_attention_lag | 29.9711 | 9 | 7.65 | -12.57 | -22.78 | 0.0550 | paper-label LINK attention-price lag over 1h, 4h, 12h, and 24h |
| VVV | Venice Token | attention_price_lag_candidate | long_attention_lag | 29.6628 | 6 | 5.98 | -9.32 | 21.61 | 0.0872 | paper-label VVV attention-price lag over 1h, 4h, 12h, and 24h |
| LAB | LAB | attention_chase_risk | wait_or_fade_watch | 28.6716 | 8 | -4.75 | 26.85 | 209.03 | 0.0134 | avoid chasing LAB; label pullback or fade setups instead |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
