# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 28 | 594900000 | 4.5000 | 0.10950000 | 638826720 | 0.00051418 | 10.0 | 32.624749 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | short | 2 | 10430000 | 30.8000 | 0.02297573 | 157905 | 0.00367955 | 3.0 | 31.508445 | wide_impact_watch | short carry aligns, but visible impact spread is wide |
| ZRO | short | 12 | 34450000 | 10.2000 | 0.11050828 | 2972259 | 0.00085054 | 5.0 | 31.467180 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| KAITO | short | 12 | 7950000 | 7.3000 | 0.10950000 | 154063 | 0.00204240 | 5.0 | 27.900167 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| EIGEN | short | 23 | 7750000 | 5.0000 | 0.10950000 | 1036763 | 0.00222717 | 5.0 | 25.301292 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 346 | 93650000 | 37.0000 | -0.02150317 | 523645 | 0.00216974 | 5.0 | 17.793393 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 23 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.937777 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 11 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.439643 | too_illiquid | perp venue volume is too low for paper priority |
| CYBER | short | 7 | 2780000 | 5.9000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 6.694467 | too_illiquid | perp venue volume is too low for paper priority |
| SOPH | short | 21 | 1210000 | 5.2000 | 0.10950000 | 27253 | 0.00294010 | 3.0 | 5.330316 | too_illiquid | perp venue volume is too low for paper priority |
| MOVE | none | 1 | 1950000 | 4.3000 | -0.23322186 | 77657 | 0.00347723 | 3.0 | -14.820569 | context_only | unlock is not large enough for a direct supply-shock ticket |
| BABY | none | 2 | 2140000 | 3.7000 | -0.73286861 | 857533 | 0.00312443 | 3.0 | -14.839087 | context_only | unlock is not large enough for a direct supply-shock ticket |
| NIL | none | 17 | 1250000 | 4.3000 | -0.23278912 | 535457 | 0.00329308 | 3.0 | -15.297140 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 3 | 1830000 | 3.8000 | -0.12661967 | 489337 | 0.00211463 | 3.0 | -15.317609 | context_only | unlock is not large enough for a direct supply-shock ticket |
| XPL | none | 18 | 7310000 | 3.4000 | 0.10950000 | 7811259 | 0.00112377 | 10.0 | -15.348651 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
