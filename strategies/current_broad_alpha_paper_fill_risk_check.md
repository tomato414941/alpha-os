# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| broad-paper-09-beat-liquidation-intensity-beat-long-liquidation-cascade-watch | BEAT | paper_long | 100.00 | 15.6225 | 0.22787608 | 15545.00500000 | 0.00643293 | -0.00000000 | 8.22787608 | 7.39461111 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
