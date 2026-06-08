# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| paper-03-inj-volume-dislocation | INJ | paper_long | 250 | 161.6753 |  |  |  | -0.00000000 | 8.00000000 | 153.67534722 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 66.6069 | 1.32927461 | 80606.36892000 | 0.01240597 | -0.12500000 | 9.32927461 | 57.15265091 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 65.5518 | 1.32899196 | 101153.63830000 | 0.00988595 | 0.25854524 | 9.32899196 | 56.48139274 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-50-hype-token-unlock | HYPE | paper_short |  | 5.3649 | 0.81386169 | 45066.62538000 |  | 0.12500000 | 8.81386169 | -3.32396761 | cost_adjusted_edge_failed | paper mark win does not survive rough spread, taker-fee, and funding haircut |
