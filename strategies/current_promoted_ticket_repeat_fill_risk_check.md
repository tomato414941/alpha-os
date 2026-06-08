# Current Paper Ticket Fill Risk Check

This checks promoted paper-ticket mark wins against rough spread, taker fee, funding, and visible-depth assumptions. It is not a live fill report.

| ticket | asset | decision | size USD | dir bps | spread | depth 10bps | usage | funding 1h | cost | net | risk action | reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| repeat-paper-05-mega-microstructure-flow | MEGA | paper_long | 100.00 | 38.1180 | 9.14476948 | 576.19495050 | 0.17355237 | 0.36800000 | 17.14476948 | 21.34118114 | depth_too_thin_for_probe | candidate size consumes too much visible 10bps depth |
