# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-lane-near-near-microstructure-flow-paper-probe | NEAR | paper_long | 100.00 | 91.9818 | 1.84145106 | 26988.93334000 | 0.00370522 | -0.12500000 | 9.84145106 | 82.01533840 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| repeat-lane-sol-sol-volume-price-dislocation | SOL | paper_long | 250 | 58.2878 | 0.30189591 | 719563.91416000 | 0.00034743 | 0.13175600 | 8.30189591 | 50.11765608 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
