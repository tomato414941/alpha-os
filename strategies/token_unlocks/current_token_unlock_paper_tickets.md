# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 28 | 594900000 | 4.5000 | 0.10950000 | 610771051 | 0.00036334 | 10.0 | 32.639833 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ZRO | short | 12 | 34450000 | 10.2000 | 0.28941638 | 3032897 | 0.00080729 | 5.0 | 31.656477 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | watch_squeeze | 2 | 10430000 | 30.8000 | -0.41118827 | 158348 | 0.00149378 | 3.0 | 29.115279 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| KAITO | short | 12 | 7950000 | 7.3000 | 0.10950000 | 154106 | 0.00185161 | 5.0 | 27.919250 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| EIGEN | short | 23 | 7750000 | 5.0000 | 0.10950000 | 1038738 | 0.00166852 | 5.0 | 25.357355 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 346 | 93650000 | 37.0000 | 0.10950000 | 526737 | 0.00206897 | 5.0 | 17.891777 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 23 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.937777 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 11 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.439643 | too_illiquid | perp venue volume is too low for paper priority |
| CYBER | short | 7 | 2780000 | 5.9000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 6.694467 | too_illiquid | perp venue volume is too low for paper priority |
| SOPH | short | 21 | 1210000 | 5.2000 | 0.10950000 | 25702 | 0.00440367 | 3.0 | 5.183803 | too_illiquid | perp venue volume is too low for paper priority |
| BABY | none | 2 | 2140000 | 3.7000 | -1.61873500 | 1308245 | 0.00263947 | 3.0 | -13.859654 | context_only | unlock is not large enough for a direct supply-shock ticket |
| MOVE | none | 1 | 1950000 | 4.3000 | 0.05697592 | 78829 | 0.00340310 | 3.0 | -14.989284 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 3 | 1830000 | 3.8000 | -0.16649782 | 490778 | 0.00217770 | 3.0 | -15.283894 | context_only | unlock is not large enough for a direct supply-shock ticket |
| NIL | none | 17 | 1250000 | 4.3000 | -0.24722735 | 529962 | 0.00330236 | 3.0 | -15.284179 | context_only | unlock is not large enough for a direct supply-shock ticket |
| XPL | none | 18 | 7310000 | 3.4000 | 0.10950000 | 7842034 | 0.00122568 | 10.0 | -15.355765 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
