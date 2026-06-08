# Current Protocol Fee Actionability

This separates fee-growth price-context screens from candidates with mature forward labels. Protocol fees are not assumed to be token-holder revenue.

| token | protocol | status | action | score | thesis | exec | labels | 4h wins | mean 4h | reason |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| UNI | Uniswap V4 | protocol_fee_pending_forward_label | wait_for_forward_label | 47.2889 | fee_decay_price_weakness_context 32.22 |  | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| SOL | Solana | protocol_fee_pending_forward_label | wait_for_forward_label | 46.4285 | fee_decay_price_weakness_context 10.71 |  | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| JUP | Jupiter Perpetual Exchange | protocol_fee_pending_forward_label | wait_for_forward_label | 44.7914 | fee_growth_price_lag_candidate 94.79 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| UNI | Uniswap V3 | protocol_fee_pending_forward_label | wait_for_forward_label | 44.1187 | fee_growth_price_lag_candidate 77.97 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| MORPHO | Morpho Blue | protocol_fee_pending_forward_label | wait_for_forward_label | 42.6136 | fee_growth_price_lag_candidate 44.03 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| PENDLE | Pendle | protocol_fee_pending_forward_label | wait_for_forward_label | 42.5217 | fee_growth_price_lag_candidate 38.04 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| CRV | Curve DEX | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 40.0639 | fee_growth_price_lag_candidate 57.00 | paper_observation_ready | 4 | 1 | -0.001081 | mature 4h labels do not support the protocol-fee thesis direction |
| AAVE | Aave V3 | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 31.2859 | fee_growth_price_lag_candidate 87.98 | paper_observation_ready | 4 | 0 | -0.021167 | mature 4h labels do not support the protocol-fee thesis direction |
