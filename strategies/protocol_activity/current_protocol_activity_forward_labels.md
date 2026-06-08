# Current Protocol Activity Forward Labels

This labels protocol activity/perp-overlap candidates with subsequent Hyperliquid returns. Positive directional return means the long activity-context direction was right before costs.

| timestamp | symbol | action | score | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-08T15:14:54.453785+00:00 | ZEC | protocol_activity_funding_overlap | 50.579739 | 0.006985 | 0.006985 | -0.001040 | -0.001040 | labeled |
| 2026-06-08T15:14:54.453785+00:00 | NEAR | protocol_activity_watch | 30.248383 | -0.013000 | -0.013000 | 0.016877 | 0.016877 | labeled |

## Interpretation

This is a short-horizon reaction label only. Protocol activity is normally slower than 15m/1h price movement, so weak labels do not fully falsify the fundamental context.
