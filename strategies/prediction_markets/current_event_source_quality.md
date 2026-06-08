# Current Event Source Quality

This checks whether event-probability paper tickets have enough fresh, source-diverse, non-duplicated external news context. It is not a probability model or trade instruction.

| question | side | sources 72h | articles 24h | newest h | unique titles | relevance | score | status | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Israel closes its airspace by June 15? | buy_no | 22 | 30 | 0.55 | 3/3 | 20.00 | 82.0000 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |
| Strait of Hormuz traffic returns to normal by end of June? | buy_yes | 25 | 23 | 0.46 | 3/3 | 6.67 | 68.6667 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |
| Strait of Hormuz traffic returns to normal by July 31? | buy_yes | 25 | 23 | 0.46 | 3/3 | 6.67 | 68.6667 | source_quality_pass | fresh multi-source news context is present and not obviously duplicated |

## Caveat

Passing this gate only means the external news feed is less obviously noisy. It does not validate truth, timing, calibration, fill quality, or adverse selection.
