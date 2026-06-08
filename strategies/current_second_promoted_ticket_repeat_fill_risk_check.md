# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-repeat-paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 4.9838 | 0.15100608 | 391483.57990000 | 0.00063860 | 0.21253600 | 8.15100608 | -2.95470507 | cost_adjusted_edge_failed | paper mark win does not survive rough spread, taker-fee, and funding haircut |
