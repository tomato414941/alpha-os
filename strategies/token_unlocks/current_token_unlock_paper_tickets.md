# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 27 | 594900000 | 4.5000 | 0.10950000 | 760191800 | 0.00017575 | 10.0 | 32.691925 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ZRO | short | 11 | 34450000 | 10.2000 | 0.10950000 | 2894196 | 0.00064664 | 5.0 | 31.512089 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | watch_squeeze | 1 | 10360000 | 30.9000 | -0.61421528 | 187325 | 0.00183885 | 3.0 | 29.419329 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| KAITO | short | 11 | 7950000 | 7.3000 | 0.10950000 | 183414 | 0.00178823 | 5.0 | 27.961851 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| EIGEN | short | 23 | 7750000 | 5.0000 | 0.10950000 | 1017780 | 0.00217984 | 5.0 | 25.304128 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 345 | 93650000 | 37.0000 | -0.25553884 | 550873 | 0.00126687 | 5.0 | 18.120440 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.971110 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.472976 | too_illiquid | perp venue volume is too low for paper priority |
| SOPH | short | 20 | 1210000 | 5.2000 | 0.10950000 | 23193 | 0.00514044 | 3.0 | 5.143208 | too_illiquid | perp venue volume is too low for paper priority |
| MOVE | none | 1 | 1930000 | 4.3000 | -7.95393896 | 357190 | 0.00424790 | 3.0 | -7.149165 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 3 | 1830000 | 3.8000 | -1.06186092 | 514430 | 0.00159063 | 3.0 | -14.327459 | context_only | unlock is not large enough for a direct supply-shock ticket |
| XPL | none | 17 | 7310000 | 3.4000 | 0.10950000 | 7529621 | 0.00027142 | 10.0 | -15.258247 | context_only | unlock is not large enough for a direct supply-shock ticket |
| NIL | none | 16 | 1250000 | 4.3000 | 0.01698476 | 555145 | 0.00181281 | 3.0 | -15.329615 | context_only | unlock is not large enough for a direct supply-shock ticket |
| BABY | none | 1 | 2270000 | 3.7000 | 0.10950000 | 1036219 | 0.00302768 | 3.0 | -15.400280 | context_only | unlock is not large enough for a direct supply-shock ticket |
| ALT | none | 16 | 1820000 | 3.9000 | -0.10952278 | 162895 | 0.00157839 | 3.0 | -15.647160 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
