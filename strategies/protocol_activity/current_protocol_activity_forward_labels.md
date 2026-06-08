# Current Protocol Activity Forward Labels

This labels protocol activity/perp-overlap candidates with subsequent Hyperliquid returns. Positive directional return means the long activity-context direction was right before costs.

| timestamp | symbol | action | score | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-08T12:22:36.092439+00:00 | ZEC | protocol_activity_funding_overlap | 44.073328 | -0.001423 | -0.001423 |  |  | labeled_15m_pending_1h |
| 2026-06-08T12:30:51.245971+00:00 | BTC | protocol_activity_watch | 35.164133 |  |  |  |  | pending_15m |
| 2026-06-08T12:30:51.245971+00:00 | NEAR | protocol_activity_watch | 28.381106 |  |  |  |  | pending_15m |

## Interpretation

This is a short-horizon reaction label only. Protocol activity is normally slower than 15m/1h price movement, so weak labels do not fully falsify the fundamental context.
