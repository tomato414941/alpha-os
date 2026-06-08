# Current Alpha Conflict Resolution Progress

This checks whether promotion-worklist dedupe and source-split items have actually been resolved. It prevents duplicate or conflicting clusters from being promoted as one trade.

| work | asset | status | score | action | dup | plans | queued | repeats | blocker | next step |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| split_conflicting_sources-06-repeat-conflict-split-required-hype-paper-long | HYPE | split_queue_ready | 528.0000 | split_lanes_before_repeat_probe | 0.0000 | 10 | 6 | 0 | lane split is queued but not yet proven by repeat outcomes | run or refresh queued lane labels for HYPE before cluster promotion |
| split_conflicting_sources-02-repeat-conflict-split-required-sol-paper-long | SOL | split_queue_ready | 525.0000 | split_lanes_before_repeat_probe | 0.6500 | 11 | 5 | 0 | lane split is queued but not yet proven by repeat outcomes | run or refresh queued lane labels for SOL before cluster promotion |
| split_conflicting_sources-04-repeat-conflict-split-required-eth-paper-long | ETH | split_plan_ready_not_queued | 430.0000 | split_lanes_before_repeat_probe | 0.5556 | 21 | 0 | 0 | lane split plan exists but is not in the active top queue | promote the highest-value ETH split lane into the active queue or lower its priority |
| dedupe_cluster-01-duplicate-dedupe-required-apt-paper-long | APT | dedupe_not_resolved | 220.0000 |  | 0.7500 | 0 | 0 | 0 | duplicate pressure remains high | choose one independent APT paper_long opportunity and suppress duplicates |
| split_conflicting_sources-01-repeat-conflict-split-required-sui-paper-long | SUI | split_not_started | 100.0000 | open_consolidated_repeat_probe | 0.5000 | 0 | 0 | 0 | cluster conflict has no active split plan yet | build lane split plan for SUI paper_long |
| split_conflicting_sources-03-repeat-conflict-split-required-link-paper-long | LINK | split_not_started | 100.0000 |  | 0.0000 | 0 | 0 | 0 | cluster conflict has no active split plan yet | build lane split plan for LINK paper_long |
| split_conflicting_sources-05-repeat-conflict-split-required-btc-paper-long | BTC | split_not_started | 100.0000 |  | 0.0000 | 0 | 0 | 0 | cluster conflict has no active split plan yet | build lane split plan for BTC paper_long |
