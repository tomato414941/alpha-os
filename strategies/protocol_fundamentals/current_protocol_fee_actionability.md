# Current Protocol Fee Actionability

This separates fee-growth price-context screens from candidates with mature forward labels. Protocol fees are not assumed to be token-holder revenue.

| token | protocol | status | action | score | thesis | exec | labels | 4h wins | mean 4h | reason |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| HYPE | Hyper Foundation HYPE Staking | protocol_fee_label_supported_watch | refresh_execution_gate | 72.9210 | fee_decay_price_weakness_context 11.29 |  | 1 | 1 | 0.012347 | at least one 4h label supports the protocol-fee thesis direction but execution or repetition is not enough |
| UNI | Uniswap V4 | protocol_fee_pending_forward_label | wait_for_forward_label | 47.2791 | fee_decay_price_weakness_context 31.98 |  | 3 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| SOL | Solana | protocol_fee_pending_forward_label | wait_for_forward_label | 46.3880 | fee_decay_price_weakness_context 9.70 |  | 3 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| UNI | Uniswap V3 | protocol_fee_pending_forward_label | wait_for_forward_label | 44.1129 | fee_growth_price_lag_candidate 77.82 | thin_volume_watch | 3 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| PENDLE | Pendle | protocol_fee_pending_forward_label | wait_for_forward_label | 42.6086 | fee_growth_price_lag_candidate 40.21 | thin_volume_watch | 3 | 0 | 0.000000 | forward label is not mature yet: pending_4h |
| AAVE | Aave V3 | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 31.2858 | fee_growth_price_lag_candidate 87.98 | paper_observation_ready | 3 | 0 | -0.021167 | mature 4h labels do not support the protocol-fee thesis direction |
| CRV | Curve DEX | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 30.1165 | fee_growth_price_lag_candidate 59.95 | wide_spread_watch | 3 | 1 | -0.001081 | mature 4h labels do not support the protocol-fee thesis direction |
| JUP | Jupiter Perpetual Exchange | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 21.8696 | fee_growth_price_lag_candidate 93.93 | thin_volume_watch | 3 | 0 | -0.014437 | mature 4h labels do not support the protocol-fee thesis direction |
| MORPHO | Morpho Blue | protocol_fee_label_failed | deprioritize_until_fresh_snapshot | 19.9292 | fee_growth_price_lag_candidate 58.56 | thin_volume_watch | 3 | 0 | -0.013408 | mature 4h labels do not support the protocol-fee thesis direction |
