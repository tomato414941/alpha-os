# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 53.1161 | 2 | 19.87 | -23.15 | -28.54 | 0.1823 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 47.0157 | 1 | 10.97 | -12.02 | -33.35 | 0.3130 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 43.3435 | 3 | 6.86 | -19.26 | -27.76 | 0.0844 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 40.1596 | 7 | 10.08 | -11.16 | 28.76 | 0.2451 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| ADA | Cardano | attention_price_lag_candidate | long_attention_lag | 36.6294 | 9 | 4.93 | -29.81 | -39.47 | 0.0941 | paper-label ADA attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 35.5087 | 6 | 5.14 | -17.75 | 37.76 | 0.0523 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 31.8051 | 4 | 3.91 | -14.44 | -21.18 | 0.0291 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |
| VVV | Venice Token | attention_price_lag_candidate | long_attention_lag | 30.3648 | 5 | 3.92 | -11.07 | 19.26 | 0.0876 | paper-label VVV attention-price lag over 1h, 4h, 12h, and 24h |
| XRP | XRP | attention_price_lag_candidate | long_attention_lag | 28.4391 | 8 | 5.64 | -13.28 | -18.62 | 0.0304 | paper-label XRP attention-price lag over 1h, 4h, 12h, and 24h |
| TON | Toncoin | attention_price_lag_candidate | long_attention_lag | 21.8161 | 10 | 3.34 | -9.81 | -32.55 | 0.0534 | paper-label TON attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
