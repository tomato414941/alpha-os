# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 27 | 594900000 | 4.5000 | 0.28021050 | 902576042 | 0.00011412 | 10.0 | 32.868799 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ZRO | short | 11 | 34450000 | 10.2000 | 0.10950000 | 2627083 | 0.00045573 | 5.0 | 31.504469 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | watch_squeeze | 1 | 10360000 | 30.9000 | -0.38971225 | 297793 | 0.00351831 | 3.0 | 29.037927 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| KAITO | short | 11 | 7950000 | 7.3000 | 0.01591955 | 194367 | 0.00190502 | 5.0 | 27.857688 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| EIGEN | short | 23 | 7750000 | 5.0000 | 0.10950000 | 1154120 | 0.00215401 | 5.0 | 25.320344 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 345 | 93650000 | 37.0000 | -0.17714560 | 605835 | 0.00180742 | 5.0 | 17.993487 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.971110 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.472976 | too_illiquid | perp venue volume is too low for paper priority |
| SOPH | short | 20 | 1210000 | 5.2000 | 0.10950000 | 25689 | 0.00323102 | 3.0 | 5.334400 | too_illiquid | perp venue volume is too low for paper priority |
| MOVE | none | 1 | 1930000 | 4.3000 | -1.73576772 | 1302468 | 0.00320558 | 3.0 | -13.168577 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 3 | 1830000 | 3.8000 | -0.44863114 | 538911 | 0.00202210 | 3.0 | -14.981388 | context_only | unlock is not large enough for a direct supply-shock ticket |
| NIL | none | 16 | 1250000 | 4.3000 | -0.37968293 | 587818 | 0.00223004 | 3.0 | -15.005373 | context_only | unlock is not large enough for a direct supply-shock ticket |
| BABY | none | 1 | 2270000 | 3.7000 | -0.32492942 | 1054140 | 0.00211535 | 3.0 | -15.091825 | context_only | unlock is not large enough for a direct supply-shock ticket |
| XPL | none | 17 | 7310000 | 3.4000 | 0.10950000 | 8363728 | 0.00101362 | 10.0 | -15.249056 | context_only | unlock is not large enough for a direct supply-shock ticket |
| LINEA | none | 2 | 2740000 | 3.6000 | 0.10950000 | 227949 | 0.00275808 | 3.0 | -15.582779 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
