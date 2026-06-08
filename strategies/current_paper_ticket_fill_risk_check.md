# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| paper-05-mega-microstructure-flow | MEGA | paper_long | 100.00 | 461.7350 | 4.26662529 | 2468.83644000 | 0.04050491 | 0.61253200 | 12.26662529 | 450.08091097 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-16-chip-repeat-execution | CHIP | paper_long | 1000 | 184.7051 | 3.17914481 | 4517.15500000 | 0.22137828 | -0.25093027 | 11.17914481 | 173.27504482 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| paper-10-arbusdt-intraday-derivatives | ARBUSDT | paper_short |  | 9.8202 |  |  |  | 0.00000000 | 8.00000000 | 1.82016817 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-11-arbusdt-intraday-derivatives | ARBUSDT | paper_short |  | 9.8202 |  |  |  | 0.00000000 | 8.00000000 | 1.82016817 | missing_execution_context | no current public execution context for the promoted paper ticket |
