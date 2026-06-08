# Current Protocol Fee Actionability

This separates fee-growth price-context screens from candidates with mature forward labels. Protocol fees are not assumed to be token-holder revenue.

| token | protocol | status | action | score | thesis | exec | labels | 4h wins | mean 4h | reason |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| CRV | Curve DEX | protocol_fee_pending_forward_label | wait_for_forward_label | 54.3623 | fee_growth_price_lag_candidate 59.06 | paper_observation_ready | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| UNI | Uniswap V4 | protocol_fee_pending_forward_label | wait_for_forward_label | 47.2761 | fee_decay_price_weakness_context 31.90 |  | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| SOL | Solana | protocol_fee_pending_forward_label | wait_for_forward_label | 46.4282 | fee_decay_price_weakness_context 10.70 |  | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| JUP | Jupiter Perpetual Exchange | protocol_fee_pending_forward_label | wait_for_forward_label | 44.8386 | fee_growth_price_lag_candidate 95.97 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| UNI | Uniswap V3 | protocol_fee_pending_forward_label | wait_for_forward_label | 44.1322 | fee_growth_price_lag_candidate 78.31 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| PENDLE | Pendle | protocol_fee_pending_forward_label | wait_for_forward_label | 42.1298 | fee_growth_price_lag_candidate 42.93 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| MORPHO | Morpho Blue | protocol_fee_pending_forward_label | wait_for_forward_label | 41.6586 | fee_growth_price_lag_candidate 54.87 | thin_volume_watch | 4 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| AAVE | Aave V3 | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 40.3859 | fee_growth_price_lag_candidate 87.98 | paper_observation_ready | 4 | 1 | -0.015667 | mature 4h labels do not support the protocol-fee thesis direction |
