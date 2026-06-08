# Current Attention Price Context

This joins CoinGecko trending attention to current price movement. It looks for attention-price lag, breakout continuation, and chase-risk candidates.

| symbol | name | status | side | score | rank | price 24h | price 7d | price 30d | vol/mcap | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ALLO | Allora | attention_chase_risk | wait_or_fade_watch | 43.0000 | 3 | 46.97 | 156.16 | 321.02 | 2.8487 | avoid chasing ALLO; label pullback or fade setups instead |
| ZEC | Zcash | attention_price_lag_candidate | long_attention_lag | 40.5447 | 2 | 1.82 | -18.15 | -25.95 | 0.1314 | paper-label ZEC attention-price lag over 1h, 4h, 12h, and 24h |
| HYPE | Hyperliquid | attention_price_lag_candidate | long_attention_lag | 38.5544 | 1 | 10.00 | -10.47 | 48.22 | 0.0617 | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| BEAT | Audiera | attention_chase_risk | wait_or_fade_watch | 35.6955 | 4 | 60.13 | 300.61 | 755.88 | 0.0739 | avoid chasing BEAT; label pullback or fade setups instead |
| SOL | Solana | attention_price_lag_candidate | long_attention_lag | 33.7759 | 6 | 3.55 | -15.88 | -28.09 | 0.0868 | paper-label SOL attention-price lag over 1h, 4h, 12h, and 24h |
| PENGU | Pudgy Penguins | attention_price_lag_candidate | long_attention_lag | 27.7078 | 9 | 3.09 | -7.62 | -34.30 | 0.2801 | paper-label PENGU attention-price lag over 1h, 4h, 12h, and 24h |
| TAO | Bittensor | attention_price_lag_candidate | long_attention_lag | 27.3930 | 7 | 1.77 | -12.52 | -30.10 | 0.0819 | paper-label TAO attention-price lag over 1h, 4h, 12h, and 24h |
| BTC | Bitcoin | attention_price_lag_candidate | long_attention_lag | 26.4607 | 5 | 3.36 | -10.65 | -20.44 | 0.0291 | paper-label BTC attention-price lag over 1h, 4h, 12h, and 24h |

## Interpretation

`attention_price_lag_candidate` means attention is high while 7d price remains weak. `attention_breakout_continuation_watch` means attention and price are already moving together. `attention_chase_risk` is a warning that the easy move may already be crowded.
