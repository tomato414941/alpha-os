# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 28 | 594900000 | 4.5000 | 0.10950000 | 766770426 | 0.00032840 | 10.0 | 32.643326 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ZRO | short | 12 | 34450000 | 10.2000 | 0.10950000 | 2902449 | 0.00089583 | 5.0 | 31.454662 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | watch_squeeze | 2 | 10360000 | 30.9000 | -0.70236716 | 187445 | 0.00235373 | 3.0 | 29.422672 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| KAITO | short | 12 | 7950000 | 7.3000 | 0.10950000 | 184036 | 0.00166419 | 5.0 | 27.940985 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| EIGEN | short | 23 | 7750000 | 5.0000 | 0.10950000 | 1010223 | 0.00272183 | 5.0 | 25.249173 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 346 | 93650000 | 37.0000 | 0.00343567 | 542917 | 0.00215620 | 5.0 | 17.778608 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 23 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.937777 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 11 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.439643 | too_illiquid | perp venue volume is too low for paper priority |
| SOPH | short | 20 | 1210000 | 5.2000 | 0.10950000 | 23223 | 0.00772343 | 3.0 | 4.884913 | too_illiquid | perp venue volume is too low for paper priority |
| MOVE | none | 1 | 1930000 | 4.3000 | -10.92133728 | 350074 | 0.00337036 | 3.0 | -4.094724 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 3 | 1830000 | 3.8000 | -1.00520825 | 510139 | 0.00232254 | 3.0 | -14.457732 | context_only | unlock is not large enough for a direct supply-shock ticket |
| NIL | none | 16 | 1250000 | 4.3000 | -0.10277670 | 552452 | 0.00246877 | 3.0 | -15.309688 | context_only | unlock is not large enough for a direct supply-shock ticket |
| BABY | none | 2 | 2270000 | 3.7000 | -0.18438924 | 1049862 | 0.00288733 | 3.0 | -15.343324 | context_only | unlock is not large enough for a direct supply-shock ticket |
| XPL | none | 17 | 7310000 | 3.4000 | 0.06033800 | 7344910 | 0.00120997 | 10.0 | -15.419735 | context_only | unlock is not large enough for a direct supply-shock ticket |
| LINEA | none | 2 | 2740000 | 3.6000 | 0.10950000 | 224247 | 0.00398248 | 3.0 | -15.705590 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
