# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 49.7407 | 2 | 7.92 | -22.58 | -29.65 | 0.1563 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 38.4684 | 1 | 77.92 | 281.50 | 751.83 | 0.0694 | avoid chasing BEAT; label pullback or fade setups instead |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 38.0000 | 8 | 62.75 | 159.95 | 354.70 | 1.9854 | avoid chasing ALLO; label pullback or fade setups instead |
| NEAR | NEAR Protocol | attention_price_lag_candidate | long_attention_lag | 35.6674 | 7 | 13.19 | -6.67 | 36.05 | 0.2301 | paper-label NEAR attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 33.6425 | 3 | 0.68 | -9.96 | -35.68 | 0.2813 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| BONK | Bonk | attention_price_lag_candidate | long_attention_lag | 33.2238 | 5 | 0.42 | -18.31 | -39.10 | 0.0698 | paper-label BONK attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 31.7926 | 6 | 3.76 | -15.18 | 40.19 | 0.0569 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 29.7200 | 10 | 1.70 | -18.05 | -29.38 | 0.0794 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 27.4696 | 4 | 1.32 | -12.80 | -21.12 | 0.0269 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
