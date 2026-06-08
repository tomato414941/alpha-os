# Current Protocol Activity Forward Labels

This labels protocol activity/perp-overlap candidates with subsequent Hyperliquid returns. Positive directional return means the long activity-context direction was right before costs.

| timestamp | symbol | action | score | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-08T11:35:09.310356+00:00 | ZEC | protocol_activity_funding_overlap | 46.917454 | -0.008346 | -0.008346 |  |  | labeled_15m_pending_1h |
| 2026-06-08T12:19:50.917603+00:00 | BTC | protocol_activity_watch | 36.164133 |  |  |  |  | pending_15m |

## Interpretation

This is a short-horizon reaction label only. Protocol activity is normally slower than 15m/1h price movement, so weak labels do not fully falsify the fundamental context.
