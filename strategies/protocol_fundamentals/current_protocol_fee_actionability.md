# Current Protocol Fee Actionability

This separates fee-growth price-context screens from candidates with mature forward labels. Protocol fees are not assumed to be token-holder revenue.

| token | protocol | status | action | score | thesis | exec | labels | 4h wins | mean 4h | reason |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| CRV | Curve DEX | protocol_fee_pending_forward_label | wait_for_forward_label | 54.2616 | fee_growth_price_lag_candidate 56.54 | paper_observation_ready | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| UNI | Uniswap V4 | protocol_fee_pending_forward_label | wait_for_forward_label | 47.2875 | fee_decay_price_weakness_context 32.19 |  | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| SOL | Solana | protocol_fee_pending_forward_label | wait_for_forward_label | 46.4284 | fee_decay_price_weakness_context 10.71 |  | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| JUP | Jupiter Perpetual Exchange | protocol_fee_pending_forward_label | wait_for_forward_label | 44.7857 | fee_growth_price_lag_candidate 94.64 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| UNI | Uniswap V3 | protocol_fee_pending_forward_label | wait_for_forward_label | 44.1097 | fee_growth_price_lag_candidate 77.74 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| PENDLE | Pendle | protocol_fee_pending_forward_label | wait_for_forward_label | 42.5044 | fee_growth_price_lag_candidate 37.61 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| MORPHO | Morpho Blue | protocol_fee_pending_forward_label | wait_for_forward_label | 41.7899 | fee_growth_price_lag_candidate 44.68 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| AAVE | Aave V3 | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 40.3859 | fee_growth_price_lag_candidate 87.98 | paper_observation_ready | 4 | 1 | -0.015667 | mature 4h labels do not support the protocol-fee thesis direction |
