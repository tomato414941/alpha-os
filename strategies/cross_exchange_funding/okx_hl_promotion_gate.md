# OKX-Hyperliquid Promotion Gate

This is a research gate, not a trade instruction. It combines fee ceiling, maker-touch proxy, and capacity so maker-only false positives do not rank above executable candidates.

| asset | action | mode | horizon | fee bps/fill/venue | headroom bps | capacity | both touch | OKX touch | HL touch | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ZEC | paper_8h_candidate | okx_cross_hl_maker | 8h | 0.25 | 0.2905 | 106210.05564167 | 0 | 0.6 | 0.2 | one-leg-cross mode survives fees and maker-leg touch gate |
| BTC | paper_8h_candidate | both_maker | 8h | 0.25 | 0.01455 | 422448.80855333 | 0.2 | 0.6 | 0.4 | maker-maker mode survives fees and same-window touch gate |
| BABY | execution_watch | both_maker | 8h | 0.25 | 0.67155 | 18182.63949194 | 0 | 0.2 | 0.2 | edge survives fees but maker-touch gate blocks the best mode |
| JTO | execution_watch | both_maker | 24h | 0.25 | 0.190775 | 54543.56750198 | 0 | 0.8 | 0.2 | edge survives fees but maker-touch gate blocks the best mode |

## Interpretation

`paper_*` means the current public-book proxy leaves fee headroom under the configured fee. `execution_watch` means the raw edge survives only through a mode whose maker leg did not pass the touch gate. `capacity_watch` means the edge may survive but size is too small for the configured threshold.
