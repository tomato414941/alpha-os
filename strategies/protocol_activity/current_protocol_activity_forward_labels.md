# Current Protocol Activity Forward Labels

This labels protocol activity/perp-overlap candidates with subsequent Hyperliquid returns. Positive directional return means the long activity-context direction was right before costs.

| timestamp | symbol | action | score | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-08T15:04:44.224453+00:00 | ZEC | protocol_activity_funding_overlap | 50.610041 |  |  |  |  | pending_15m |
| 2026-06-08T15:11:24.272912+00:00 | BTC | protocol_activity_watch | 34.156379 |  |  |  |  | pending_15m |

## Interpretation

This is a short-horizon reaction label only. Protocol activity is normally slower than 15m/1h price movement, so weak labels do not fully falsify the fundamental context.
