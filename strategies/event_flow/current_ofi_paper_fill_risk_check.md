# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ofi-paper-01-eth-short | ETH | paper_short | 100 | 96.3954 | 0.58995310 | 12405452.08724500 | 0.00000806 | -0.12891200 | 8.58995310 | 87.67653129 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| ofi-paper-02-sui-short | SUI | paper_short | 100 | 84.3970 | 1.83888721 | 72465.29272500 | 0.00137997 | 0.12500000 | 9.83888721 | 74.68309117 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| ofi-paper-03-bnb-short | BNB | paper_short | 100 | 50.6898 | 1.15424887 | 95934.50999500 | 0.00104238 | 0.03705700 | 9.15424887 | 41.57261603 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
