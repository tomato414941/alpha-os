# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ofi-repeat-02-sui-short | SUI | paper_short | 100 | 31.7869 | 0.13178616 | 82640.39022300 | 0.00121006 | 0.12500000 | 8.13178616 | 23.78011028 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| ofi-repeat-01-bnb-short | BNB | paper_short | 100 | 20.7749 | 1.65270134 | 82038.41595000 | 0.00121894 | 0.07317000 | 9.65270134 | 11.19540683 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| ofi-repeat-03-eth-short | ETH | paper_short | 100 | 15.3374 | 0.59096416 | 12655836.39625500 | 0.00000790 | -0.13627400 | 8.59096416 | 6.61018515 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
