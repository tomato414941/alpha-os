# Current Protocol Activity Forward Labels

This labels protocol activity/perp-overlap candidates with subsequent Hyperliquid returns. Positive directional return means the long activity-context direction was right before costs.

| timestamp | symbol | action | score | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-07T18:28:52.395358+00:00 | LINK | protocol_activity_watch | 33.229618 | -0.001022 | -0.001022 | -0.018377 | -0.018377 | labeled |
| 2026-06-07T18:28:52.395358+00:00 | ZEC | protocol_activity_funding_overlap | 40.564912 | -0.007085 | -0.007085 | -0.018694 | -0.018694 | labeled |
| 2026-06-08T00:40:58.931225+00:00 | BTC | protocol_activity_watch | 37.164986 |  |  |  |  | pending_15m |
| 2026-06-08T00:40:58.931225+00:00 | NEAR | protocol_activity_watch | 30.435058 |  |  |  |  | pending_15m |

## Interpretation

This is a short-horizon reaction label only. Protocol activity is normally slower than 15m/1h price movement, so weak labels do not fully falsify the fundamental context.
