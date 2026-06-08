# Current Event Source Quality

This checks whether event-probability paper tickets have enough fresh, source-diverse, non-duplicated external news context. It is not a probability model or trade instruction.

| question | side | sources 72h | articles 24h | newest h | unique titles | relevance | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Strait of Hormuz traffic returns to normal by July 31? | buy_no | 25 | 23 | 0.93 | 3/3 | 6.67 | 68.6667 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | buy_yes | 8 | 4 | 2.75 | 3/3 | 20.00 | 61.2000 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |
| Will Roberto Sánchez Palomino win the 2026 Peruvian presidential election? | buy_no | 1 | 1 | 13.51 | 3/3 | 0.00 | 18.8000 | source_quality_fail | too few independent sources |

## Caveat

Passing this gate only means the external news feed is less obviously noisy. It does not validate truth, timing, calibration, fill quality, or adverse selection.
