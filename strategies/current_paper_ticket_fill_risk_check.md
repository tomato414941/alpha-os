# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| paper-50-hype-token-unlock | HYPE | paper_short |  | 5.3649 | 0.81386169 | 45066.62538000 |  | 0.12500000 | 8.81386169 | -3.32396761 | cost_adjusted_edge_failed | paper mark win does not survive rough spread, taker-fee, and funding haircut |
