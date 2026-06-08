# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| paper-05-mega-microstructure-flow | MEGA | paper_long | 100.00 | 373.7275 | 11.32945267 | 2314.63432200 | 0.04320337 | 0.75779900 | 19.32945267 | 355.15588820 | cost_adjusted_paper_probe | paper mark win survives rough spread, taker-fee, funding, and visible-depth checks |
| paper-16-chip-repeat-execution | CHIP | paper_long | 1000 | 210.6286 | 3.17208565 | 5220.51300000 | 0.19155206 | -0.49348296 | 11.17208565 | 198.96307689 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
| paper-10-arbusdt-intraday-derivatives | ARBUSDT | paper_short |  | 27.0055 |  |  |  | 0.00000000 | 8.00000000 | 19.00546247 | missing_execution_context | no current public execution context for the promoted paper ticket |
| paper-11-arbusdt-intraday-derivatives | ARBUSDT | paper_short |  | 27.0055 |  |  |  | 0.00000000 | 8.00000000 | 19.00546247 | missing_execution_context | no current public execution context for the promoted paper ticket |
