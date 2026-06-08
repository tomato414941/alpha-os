# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 27 | 594900000 | 4.5000 | 0.10950000 | 881357291 | 0.00012984 | 10.0 | 32.696516 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ZRO | short | 11 | 34450000 | 10.2000 | 0.10950000 | 2604693 | 0.00028373 | 5.0 | 31.519430 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | watch_squeeze | 1 | 10360000 | 30.9000 | -0.95318962 | 298745 | 0.00240770 | 3.0 | 29.712560 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| KAITO | short | 11 | 7950000 | 7.3000 | 0.10950000 | 195800 | 0.00207866 | 5.0 | 27.934047 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| EIGEN | short | 23 | 7750000 | 5.0000 | 0.10950000 | 1210527 | 0.00162690 | 5.0 | 25.378696 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 345 | 93650000 | 37.0000 | 0.01800443 | 595639 | 0.00213816 | 5.0 | 17.800252 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.971110 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.472976 | too_illiquid | perp venue volume is too low for paper priority |
| SOPH | short | 20 | 1210000 | 5.2000 | 0.10950000 | 27826 | 0.00361337 | 3.0 | 5.296379 | too_illiquid | perp venue volume is too low for paper priority |
| MOVE | none | 1 | 1930000 | 4.3000 | -2.87776512 | 1226734 | 0.00270289 | 3.0 | -11.983884 | context_only | unlock is not large enough for a direct supply-shock ticket |
| BABY | none | 1 | 2270000 | 3.7000 | -0.50522950 | 1057278 | 0.00220153 | 3.0 | -14.919830 | context_only | unlock is not large enough for a direct supply-shock ticket |
| XPL | none | 17 | 7310000 | 3.4000 | 0.10950000 | 8644983 | 0.00078793 | 10.0 | -15.198362 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 3 | 1830000 | 3.8000 | 0.07317316 | 604067 | 0.00167273 | 3.0 | -15.315393 | context_only | unlock is not large enough for a direct supply-shock ticket |
| NIL | none | 16 | 1250000 | 4.3000 | -0.03011863 | 582393 | 0.00292564 | 3.0 | -15.425040 | context_only | unlock is not large enough for a direct supply-shock ticket |
| ALT | none | 16 | 1820000 | 3.9000 | -0.22407642 | 174905 | 0.00191238 | 3.0 | -15.564804 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
