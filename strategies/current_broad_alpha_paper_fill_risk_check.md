# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| broad-paper-pol-paper-short | POL | paper_short | 100 | 298.2586 | 4.45510842 | 5698.14415650 | 0.01754957 | 0.12500000 | 12.45510842 | 285.92850588 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-beat-paper-long | BEAT | paper_long | 100.00 | 50.0839 | 0.22787608 | 15545.00500000 | 0.00643293 | -0.00000000 | 8.22787608 | 41.85597992 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
