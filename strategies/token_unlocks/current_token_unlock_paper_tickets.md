# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 28 | 594900000 | 4.5000 | 0.10950000 | 603221747 | 0.00013295 | 10.0 | 32.662872 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ZRO | short | 12 | 34450000 | 10.2000 | 0.25601012 | 2059313 | 0.00045507 | 5.0 | 31.560935 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | watch_squeeze | 2 | 10430000 | 30.8000 | -0.37705405 | 152246 | 0.00217173 | 3.0 | 29.012739 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| KAITO | short | 12 | 7950000 | 7.3000 | 0.10880971 | 116370 | 0.00204339 | 5.0 | 27.895607 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| EIGEN | short | 23 | 7750000 | 5.0000 | 0.10950000 | 868327 | 0.00282326 | 5.0 | 25.224840 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 346 | 93650000 | 37.0000 | 0.10950000 | 509799 | 0.00251344 | 5.0 | 17.845636 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 23 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.937777 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 11 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.439643 | too_illiquid | perp venue volume is too low for paper priority |
| CYBER | short | 7 | 2780000 | 5.9000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 6.694467 | too_illiquid | perp venue volume is too low for paper priority |
| SOPH | short | 21 | 1210000 | 5.2000 | 0.10950000 | 26080 | 0.00390190 | 3.0 | 5.234018 | too_illiquid | perp venue volume is too low for paper priority |
| NIL | none | 17 | 1250000 | 4.3000 | -0.83217284 | 492914 | 0.00235288 | 3.0 | -14.607991 | context_only | unlock is not large enough for a direct supply-shock ticket |
| MOVE | none | 2 | 1950000 | 4.3000 | 0.10950000 | 69559 | 0.00255798 | 3.0 | -14.886509 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 4 | 1830000 | 3.8000 | -0.50639545 | 401876 | 0.00281467 | 3.0 | -15.049917 | context_only | unlock is not large enough for a direct supply-shock ticket |
| ALT | none | 17 | 1820000 | 3.9000 | -0.72024370 | 136558 | 0.00141743 | 3.0 | -15.056311 | context_only | unlock is not large enough for a direct supply-shock ticket |
| BABY | none | 2 | 2140000 | 3.7000 | -0.06606179 | 1667569 | 0.00195491 | 3.0 | -15.307939 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
