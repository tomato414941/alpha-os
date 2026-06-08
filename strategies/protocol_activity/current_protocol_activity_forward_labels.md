# Current Protocol Activity Forward Labels

This labels protocol activity/perp-overlap candidates with subsequent Hyperliquid returns. Positive directional return means the long activity-context direction was right before costs.

| timestamp | symbol | action | score | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-08T12:37:00.058573+00:00 | ZEC | protocol_activity_funding_overlap | 47.380076 | 0.006973 | 0.006973 | 0.016952 | 0.016952 | labeled |
| 2026-06-08T15:01:16.662718+00:00 | BTC | protocol_activity_watch | 35.151158 |  |  |  |  | pending_15m |

## Interpretation

This is a short-horizon reaction label only. Protocol activity is normally slower than 15m/1h price movement, so weak labels do not fully falsify the fundamental context.
