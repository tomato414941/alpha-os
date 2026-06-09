# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 27 | 594900000 | 4.5000 | 0.10950000 | 1025397896 | 0.00032092 | 10.0 | 32.677408 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ZRO | short | 11 | 34450000 | 10.2000 | 0.10950000 | 2473759 | 0.00056768 | 5.0 | 31.477941 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | watch_squeeze | 1 | 10360000 | 30.9000 | -0.54292027 | 389458 | 0.00264419 | 3.0 | 29.287714 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| EIGEN | short | 22 | 7750000 | 5.0000 | 0.01434800 | 1494386 | 0.00165107 | 5.0 | 25.342846 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 345 | 93650000 | 37.0000 | -0.23820893 | 626965 | 0.00217787 | 5.0 | 18.019618 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.971110 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.472976 | too_illiquid | perp venue volume is too low for paper priority |
| KAITO | short | 11 | 7950000 | 7.3000 | 0.10950000 | 169114 | 0.00315814 | 5.0 | 7.823430 | wide_impact_watch | short carry aligns, but visible impact spread is wide |
| SOPH | short | 20 | 1210000 | 5.2000 | 0.10950000 | 26552 | 0.00860456 | 3.0 | 4.797132 | too_illiquid | perp venue volume is too low for paper priority |
| MOVE | none | 0 | 1930000 | 4.3000 | -1.48420417 | 1717949 | 0.00415216 | 3.0 | -13.439917 | context_only | unlock is not large enough for a direct supply-shock ticket |
| BABY | none | 1 | 2270000 | 3.7000 | -0.78767642 | 833518 | 0.00296776 | 3.0 | -14.736381 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 2 | 1830000 | 3.8000 | -0.61704914 | 782814 | 0.00203437 | 3.0 | -14.756474 | context_only | unlock is not large enough for a direct supply-shock ticket |
| NIL | none | 16 | 1250000 | 4.3000 | -0.27067524 | 547853 | 0.00206191 | 3.0 | -15.101564 | context_only | unlock is not large enough for a direct supply-shock ticket |
| XPL | none | 17 | 7310000 | 3.4000 | 0.00181682 | 7219405 | 0.00102668 | 10.0 | -15.472477 | context_only | unlock is not large enough for a direct supply-shock ticket |
| LINEA | none | 1 | 2740000 | 3.6000 | 0.10950000 | 288025 | 0.00371594 | 3.0 | -15.639225 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
