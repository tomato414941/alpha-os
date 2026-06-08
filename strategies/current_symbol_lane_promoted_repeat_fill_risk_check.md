# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-lane-hype-hype-protocol-fee-growth | HYPE | paper_short | 100 | 6.2043 | 0.81725386 | 50059.10871000 | 0.00199764 | 0.12500000 | 8.81725386 | -2.48796964 | cost_adjusted_edge_failed | paper mark win does not survive rough spread, taker-fee, and funding haircut |
| repeat-lane-hype-hype-unlock-actionability | HYPE | paper_short | 100 | 6.2043 | 0.81725386 | 50059.10871000 | 0.00199764 | 0.12500000 | 8.81725386 | -2.48796964 | cost_adjusted_edge_failed | paper mark win does not survive rough spread, taker-fee, and funding haircut |
