# Current Token Unlock Paper Tickets

This converts current token unlock/perp overlaps into paper tickets. It is not a live trade instruction.

| symbol | side | in | value USD | % supply | funding | volume USD | impact | max lev | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HYPE | short | 27 | 594900000 | 4.5000 | 0.10950000 | 1042653110 | 0.00001575 | 10.0 | 32.707925 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ZRO | short | 11 | 34450000 | 10.2000 | 0.10950000 | 2830814 | 0.00041077 | 5.0 | 31.529338 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| ME | watch_squeeze | 1 | 10360000 | 30.9000 | -0.14802823 | 347465 | 0.00241702 | 3.0 | 28.911340 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| EIGEN | short | 22 | 7750000 | 5.0000 | 0.10950000 | 1128644 | 0.00163221 | 5.0 | 25.403310 | paper_short_candidate | supply shock and short carry align on a tradable perp venue |
| PYTH | none | 345 | 93650000 | 37.0000 | -0.33543004 | 623401 | 0.00155768 | 5.0 | 18.178502 | context_only | unlock is not large enough for a direct supply-shock ticket |
| AI | short | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.971110 | too_illiquid | perp venue volume is too low for paper priority |
| PIXEL | short | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | 3.0 | 12.472976 | too_illiquid | perp venue volume is too low for paper priority |
| SOPH | short | 20 | 1210000 | 5.2000 | 0.10950000 | 25583 | 0.00486662 | 3.0 | 5.170830 | too_illiquid | perp venue volume is too low for paper priority |
| KAITO | watch_squeeze | 11 | 7950000 | 7.3000 | -0.07434612 | 189701 | 0.00178706 | 5.0 | 4.927443 | crowded_short_risk | supply shock overlaps negative funding, so new shorts may be crowded |
| MOVE | none | 1 | 1930000 | 4.3000 | -1.01632732 | 1498636 | 0.00172687 | 3.0 | -13.720529 | context_only | unlock is not large enough for a direct supply-shock ticket |
| IO | none | 3 | 1830000 | 3.8000 | -1.41607064 | 680210 | 0.00247934 | 3.0 | -14.045542 | context_only | unlock is not large enough for a direct supply-shock ticket |
| XPL | none | 17 | 7310000 | 3.4000 | 0.10950000 | 7975539 | 0.00119454 | 10.0 | -15.305967 | context_only | unlock is not large enough for a direct supply-shock ticket |
| BABY | none | 1 | 2270000 | 3.7000 | -0.18913190 | 942172 | 0.00296717 | 3.0 | -15.324001 | context_only | unlock is not large enough for a direct supply-shock ticket |
| ALT | none | 16 | 1820000 | 3.9000 | -0.42319122 | 184497 | 0.00153557 | 3.0 | -15.327050 | context_only | unlock is not large enough for a direct supply-shock ticket |
| NIL | none | 16 | 1250000 | 4.3000 | -0.14047536 | 585772 | 0.00333098 | 3.0 | -15.354879 | context_only | unlock is not large enough for a direct supply-shock ticket |

## Caveat

Unlock tickets need event-window labels, venue depth, borrow/funding persistence, and stop logic. A supply shock with negative funding is treated as crowded-short risk, not as an automatic short.
