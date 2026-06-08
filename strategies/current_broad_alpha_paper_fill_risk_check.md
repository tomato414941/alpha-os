# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| broad-paper-pol-paper-short | POL | paper_short | 100 | 230.4557 | 4.93268154 | 4980.82630650 | 0.02007699 | -0.00234900 | 12.93268154 | 217.52069380 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-zro-paper-short | ZRO | paper_short | 100 | 216.0244 | 5.57899508 | 582.47049000 | 0.17168252 | 0.23421800 | 13.57899508 | 202.67959841 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| broad-paper-zec-paper-long | ZEC | paper_long | 100 | 215.1219 | 2.84261739 | 115191.02100000 | 0.00086812 | 1.16010900 | 10.84261739 | 205.43935760 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-chip-paper-long | CHIP | paper_long | 100 | 65.8051 | 6.19880452 | 4040.77269000 | 0.02474774 | -0.12500000 | 14.19880452 | 51.48134079 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-eigen-paper-short | EIGEN | paper_short | 100 | 48.9663 | 5.44810678 | 1388.01244200 | 0.07204546 | 0.12500000 | 13.44810678 | 35.64316090 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-hype-paper-long | HYPE | paper_long | 100 | 40.4699 | 0.15686151 | 84402.47447500 | 0.00118480 | -0.12500000 | 8.15686151 | 32.18803015 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-beat-paper-long | BEAT | paper_long | 100.00 | 25.2717 | 0.22787608 | 15545.00500000 | 0.00643293 | -0.00000000 | 8.22787608 | 17.04379438 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| broad-paper-btc-paper-short | BTC | paper_short | 100 | 19.8456 | 0.15778968 | 4823105.60306500 | 0.00002073 | 0.11265300 | 8.15778968 | 11.80050830 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
