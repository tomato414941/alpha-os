# Current Paper Tickets

These are current paper-observation tickets opened from the cross-lane probe plan. They are not trade instructions and do not imply live execution.

| ticket | rank | opportunity | side | asset | venue | size USD | entry mark | checkpoints | decision | required record |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| ofi-paper-01-eth-short | 1 | ofi_execution_survival:imbalance_1pct/bottom_20/paper_short:short_horizon_maker_probe_only | short | ETH |  | 100 | 1711.700000000000 | 5m,15m | paper_short | 5m/15m mark move, spread/depth, maker fill assumption, queue/cancel note, adverse selection |
| ofi-paper-02-sui-short | 2 | ofi_execution_survival:imbalance_1pct/bottom_20/paper_short:short_horizon_maker_probe_only | short | SUI |  | 100 | 0.767800000000 | 5m,15m | paper_short | 5m/15m mark move, spread/depth, maker fill assumption, queue/cancel note, adverse selection |
| ofi-paper-03-bnb-short | 3 | ofi_execution_survival:imbalance_1pct/bottom_20/paper_short:short_horizon_maker_probe_only | short | BNB |  | 100 | 609.590000000000 | 5m,15m | paper_short | 5m/15m mark move, spread/depth, maker fill assumption, queue/cancel note, adverse selection |

## Rule

A ticket can only promote a candidate after the checkpoint record includes mark movement, spread or fill assumption, funding where relevant, and stop or adverse-excursion notes. Missing entry marks are allowed for non-perp or externally quoted candidates, but they must be filled before promotion.
