# Current Protocol Activity Forward Labels

This labels protocol activity/perp-overlap candidates with subsequent Hyperliquid returns. Positive directional return means the long activity-context direction was right before costs.

| timestamp | symbol | action | score | raw 15m | dir 15m | raw 1h | dir 1h | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-07T18:28:52.585752+00:00 | BTC | protocol_activity_watch | 32.151710 | -0.000563 | -0.000563 |  |  | labeled_15m_pending_1h |
| 2026-06-07T18:28:52.585752+00:00 | SUI | protocol_activity_watch | 33.065420 | -0.001989 | -0.001989 |  |  | labeled_15m_pending_1h |
| 2026-06-07T18:28:52.585752+00:00 | APT | protocol_activity_watch | 28.529689 | -0.002542 | -0.002542 |  |  | labeled_15m_pending_1h |
| 2026-06-07T18:28:52.395358+00:00 | ZEC | protocol_activity_funding_overlap | 39.549468 | -0.007085 | -0.007085 |  |  | labeled_15m_pending_1h |
| 2026-06-07T18:28:52.585752+00:00 | NEAR | protocol_activity_watch | 26.436771 | -0.010382 | -0.010382 |  |  | labeled_15m_pending_1h |
| 2026-06-07T18:28:52.585752+00:00 | TON | protocol_activity_watch | 26.150774 | -0.012347 | -0.012347 |  |  | labeled_15m_pending_1h |

## Interpretation

This is a short-horizon reaction label only. Protocol activity is normally slower than 15m/1h price movement, so weak labels do not fully falsify the fundamental context.
