# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| paper-05-mega-microstructure-flow | MEGA | paper_long | 100.00 | 268.2808 | 11.06544420 | 1122.83969600 | 0.08905991 | 0.09460500 | 19.06544420 | 249.30997355 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-16-chip-repeat-execution | CHIP | paper_long | 1000 | 223.5904 | 3.16906988 | 5191.00100000 | 0.19264107 | 0.57389348 | 11.16906988 | 212.99523190 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| paper-03-inj-volume-dislocation | INJ | paper_long | 250 | 150.1013 |  |  |  | -0.00000000 | 8.00000000 | 142.10127315 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-27-pepe-liquidation-intensity | PEPE | paper_long | 100.00 | 86.6113 | 3.57717761 | 114123.51700000 | 0.00087624 | -0.49557445 | 11.57717761 | 74.53857959 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-28-doge-liquidation-intensity | DOGE | paper_long | 100.00 | 76.4776 |  |  |  | -0.00000000 | 8.00000000 | 68.47756775 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-26-sui-liquidation-intensity | SUI | paper_long | 100.00 | 73.6574 | 1.32934530 | 97508.84550000 | 0.00102555 | 0.26157897 | 9.32934530 | 64.58965968 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-09-sol-volume-dislocation | SOL | paper_long | 250 | 72.7140 | 0.15102888 | 691488.91950000 | 0.00036154 | 0.13408400 | 8.15102888 | 64.69705180 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-01-sui-repeat-execution | SUI | paper_long | 1000 | 67.6769 | 1.46168719 | 52473.70350700 | 0.01905716 | -0.12500000 | 9.46168719 | 58.09022910 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-19-sui-microstructure-flow | SUI | paper_long | 100.00 | 67.6769 | 1.46168719 | 52473.70350700 | 0.00190572 | -0.12500000 | 9.46168719 | 58.09022910 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-02-sui-repeat-execution | SUI | paper_long | 1000 | 62.8763 | 1.32934530 | 97508.84550000 | 0.01025548 | 0.26157897 | 9.32934530 | 53.80848785 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-12-virtual-volume-dislocation | VIRTUAL | paper_long | 250 | 37.2420 |  |  |  | -0.00000000 | 8.00000000 | 29.24202141 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-07-fartcoin-volume-dislocation | FARTCOIN | paper_long | 250 | 33.8689 |  |  |  | -0.00000000 | 8.00000000 | 25.86886670 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-20-mon-microstructure-flow | MON | paper_long | 100.00 | 16.0433 | 3.20388127 | 2256.81895900 | 0.04431016 | 0.06850600 | 11.20388127 | 4.90789572 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-15-link-volume-dislocation | LINK | paper_long | 250 | 6.8356 |  |  |  | -0.00000000 | 8.00000000 | -1.16438391 | missing_execution_context | no current public execution context for the promoted paper ticket |
