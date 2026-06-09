# Current Event Source Quality

This checks whether event-probability paper tickets have enough fresh, source-diverse, non-duplicated external news context. It is not a probability model or trade instruction.

| question | side | sources 72h | articles 24h | newest h | unique titles | relevance | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Israel closes its airspace by June 15? | buy_yes | 17 | 24 | 6.11 | 3/3 | 20.00 | 76.0000 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |
| US x Iran permanent peace deal by June 15, 2026? | buy_yes | 29 | 24 | 0.61 | 3/3 | 13.33 | 75.3333 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |
| Strait of Hormuz traffic returns to normal by end of June? | buy_yes | 26 | 29 | 0.91 | 3/3 | 13.33 | 75.3333 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |
| Will Keiko Fujimori win the 2026 Peruvian presidential election? | buy_no | 6 | 3 | 2.73 | 3/3 | 13.33 | 49.7333 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |

## Caveat

Passing this gate only means the external news feed is less obviously noisy. It does not validate truth, timing, calibration, fill quality, or adverse selection.
