# Current Protocol Fee Actionability

This separates fee-growth price-context screens from candidates with mature forward labels. Protocol fees are not assumed to be token-holder revenue.

| token | protocol | status | action | score | thesis | exec | labels | 4h wins | mean 4h | reason |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| UNI | Uniswap V4 | protocol_fee_pending_forward_label | wait_for_forward_label | 47.3096 | fee_decay_price_weakness_context 32.74 |  | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| SOL | Solana | protocol_fee_pending_forward_label | wait_for_forward_label | 46.4294 | fee_decay_price_weakness_context 10.73 |  | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| UNI | Uniswap V3 | protocol_fee_pending_forward_label | wait_for_forward_label | 44.2442 | fee_growth_price_lag_candidate 81.10 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| PENDLE | Pendle | protocol_fee_pending_forward_label | wait_for_forward_label | 42.5763 | fee_growth_price_lag_candidate 39.41 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| CRV | Curve DEX | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 40.0903 | fee_growth_price_lag_candidate 57.66 | paper_observation_ready | 4 | 1 | -0.001081 | mature 4h labels do not support the protocol-fee thesis direction |
| AAVE | Aave V3 | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 31.2859 | fee_growth_price_lag_candidate 87.98 | paper_observation_ready | 4 | 0 | -0.021167 | mature 4h labels do not support the protocol-fee thesis direction |
| JUP | Jupiter Perpetual Exchange | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 21.9844 | fee_growth_price_lag_candidate 96.80 | thin_volume_watch | 4 | 0 | -0.014437 | mature 4h labels do not support the protocol-fee thesis direction |
| MORPHO | Morpho Blue | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 19.7283 | fee_growth_price_lag_candidate 51.88 | thin_volume_watch | 4 | 0 | -0.013408 | mature 4h labels do not support the protocol-fee thesis direction |
