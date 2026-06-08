# Current Paper Tickets

These are current paper-observation tickets opened from the cross-lane probe plan. They are not trade instructions and do not imply live execution.

| ticket | rank | opportunity | side | asset | venue | size USD | entry mark | checkpoints | decision | required record |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| ofi-repeat-01-bnb-short | 1 | ofi_execution_survival_repeat:broad-fill-audit-bnb-short-50bps-stop:15m:ofi-paper-03-bnb-short | short | BNB |  | 100 | 606.500000000000 | 5m,15m | paper_short | fresh 5m/15m mark move, cost, stop status, queue/cancel note, adverse selection |
| ofi-repeat-02-sui-short | 2 | ofi_execution_survival_repeat:broad-fill-audit-sui-short-50bps-stop:15m:ofi-paper-02-sui-short | short | SUI |  | 100 | 0.761320000000 | 5m,15m | paper_short | fresh 5m/15m mark move, cost, stop status, queue/cancel note, adverse selection |
| ofi-repeat-03-eth-short | 3 | ofi_execution_survival_repeat:broad-fill-audit-eth-short-50bps-stop:15m:ofi-paper-01-eth-short | short | ETH |  | 100 | 1695.200000000000 | 5m,15m | paper_short | fresh 5m/15m mark move, cost, stop status, queue/cancel note, adverse selection |

## Rule

A ticket can only promote a candidate after the checkpoint record includes mark movement, spread or fill assumption, funding where relevant, and stop or adverse-excursion notes. Missing entry marks are allowed for non-perp or externally quoted candidates, but they must be filled before promotion.
