# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| broad-paper-pol-paper-short | POL | paper_short | 100 | 230.4557 | 2.27565804 | 5651.94759000 | 0.01769302 | 0.03026100 | 10.27565804 | 220.21032730 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-zro-paper-short | ZRO | paper_short | 100 | 216.0244 |  |  |  | 0.00000000 | 8.00000000 | 208.02437549 | missing_execution_context | no current public execution context for the promoted paper ticket |
| broad-paper-zec-paper-long | ZEC | paper_long | 100 | 215.1219 | 2.84261739 | 115191.02100000 | 0.00086812 | 1.16010900 | 10.84261739 | 205.43935760 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-chip-paper-long | CHIP | paper_long | 100 | 65.8051 | 3.25400464 | 3567.72693000 | 0.02802905 | -0.12500000 | 11.25400464 | 54.42614067 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-eigen-paper-short | EIGEN | paper_short | 100 | 48.9663 | 5.44810678 | 1388.01244200 | 0.07204546 | 0.12500000 | 13.44810678 | 35.64316090 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-hype-paper-long | HYPE | paper_long | 100 | 40.4699 | 0.15685905 | 65655.75730500 | 0.00152310 | -0.12500000 | 8.15685905 | 32.18803261 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-beat-paper-long | BEAT | paper_long | 100.00 | 25.2717 | 0.22787608 | 15545.00500000 | 0.00643293 | -0.00000000 | 8.22787608 | 17.04379438 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-btc-paper-short | BTC | paper_short | 100 | 19.8456 | 0.15782455 | 1354031.45331000 | 0.00007385 | 0.11386000 | 8.15782455 | 11.80168043 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
