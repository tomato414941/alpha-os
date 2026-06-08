# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| lane-hype-hype-volume-price-dislocation | HYPE | paper_long | 250 | 14.9339 | 0.97216371 | 144113.99872000 | 0.00173474 | -0.12500000 | 8.97216371 | 5.83668906 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| lane-hype-hype-microstructure-flow-paper-probe | HYPE | paper_long | 100.00 | 14.9339 | 0.97216371 | 144113.99872000 | 0.00069390 | -0.12500000 | 8.97216371 | 5.83668906 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
