# Current Alpha Conflict Resolution Progress

This checks whether promotion-worklist dedupe and source-split items have actually been resolved. It prevents duplicate or conflicting clusters from being promoted as one trade.

| work | asset | status | score | action | dup | plans | queued | repeats | blocker | next step |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| split_conflicting_sources-01-repeat-conflict-split-required-zec-paper-long | ZEC | split_queue_ready | 567.0000 | split_lanes_before_repeat_probe | 0.4444 | 13 | 9 | 0 | lane split is queued but not yet proven by repeat outcomes | run or refresh queued lane labels for ZEC before cluster promotion |
| split_conflicting_sources-02-repeat-conflict-split-required-sol-paper-long | SOL | split_queue_ready | 525.0000 | split_lanes_before_repeat_probe | 0.6500 | 11 | 5 | 0 | lane split is queued but not yet proven by repeat outcomes | run or refresh queued lane labels for SOL before cluster promotion |
| split_conflicting_sources-03-repeat-conflict-split-required-btc-paper-long | BTC | split_plan_ready_not_queued | 430.0000 | split_lanes_before_repeat_probe | 0.6000 | 22 | 0 | 0 | lane split plan exists but is not in the active top queue | promote the highest-value BTC split lane into the active queue or lower its priority |
| split_conflicting_sources-05-repeat-conflict-split-required-eth-paper-long | ETH | split_plan_ready_not_queued | 430.0000 | split_lanes_before_repeat_probe | 0.5294 | 21 | 0 | 0 | lane split plan exists but is not in the active top queue | promote the highest-value ETH split lane into the active queue or lower its priority |
| dedupe_cluster-01-duplicate-dedupe-required-sui-paper-long | SUI | dedupe_conflicts_with_consolidated_repeat | 260.0000 | open_consolidated_repeat_probe | 0.5556 | 0 | 0 | 0 | dedupe work says do not reuse the same move, but cluster plan still opens one consolidated repeat | rewrite SUI paper_long repeat work as unique-opportunity dedupe before any consolidated repeat |
| split_conflicting_sources-04-repeat-conflict-split-required-mon-paper-long | MON | split_not_started | 100.0000 | open_consolidated_repeat_probe | 0.0000 | 0 | 0 | 0 | cluster conflict has no active split plan yet | build lane split plan for MON paper_long |
| split_conflicting_sources-06-repeat-conflict-split-required-apt-paper-long | APT | split_not_started | 100.0000 |  | 0.0000 | 0 | 0 | 0 | cluster conflict has no active split plan yet | build lane split plan for APT paper_long |
