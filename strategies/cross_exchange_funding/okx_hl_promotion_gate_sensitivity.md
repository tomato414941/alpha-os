# OKX-Hyperliquid Promotion Gate Sensitivity

This sweeps account fee assumptions through the promotion gate. It is a research sensitivity table, not a trade instruction.

| fee bps/fill/venue | asset | action | mode | horizon | headroom bps | capacity | both touch | OKX touch | HL touch |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | ZEC | paper_8h_candidate | okx_cross_hl_maker | 8h | 0.4405 | 106210.05564167 | 0 | 0.6 | 0.2 |
| 0.1 | BTC | paper_8h_candidate | both_maker | 8h | 0.16455 | 422448.80855333 | 0.2 | 0.6 | 0.4 |
| 0.1 | BABY | execution_watch | both_maker | 8h | 0.82155 | 18182.63949194 | 0 | 0.2 | 0.2 |
| 0.1 | JTO | execution_watch | both_maker | 8h | 0.046925 | 54543.56750198 | 0 | 0.8 | 0.2 |
| 0.25 | ZEC | paper_8h_candidate | okx_cross_hl_maker | 8h | 0.2905 | 106210.05564167 | 0 | 0.6 | 0.2 |
| 0.25 | BTC | paper_8h_candidate | both_maker | 8h | 0.01455 | 422448.80855333 | 0.2 | 0.6 | 0.4 |
| 0.25 | BABY | execution_watch | both_maker | 8h | 0.67155 | 18182.63949194 | 0 | 0.2 | 0.2 |
| 0.25 | JTO | execution_watch | both_maker | 24h | 0.190775 | 54543.56750198 | 0 | 0.8 | 0.2 |
| 0.5 | ZEC | paper_8h_candidate | okx_cross_hl_maker | 8h | 0.0405 | 106210.05564167 | 0 | 0.6 | 0.2 |
| 0.5 | BTC | paper_24h_candidate | both_maker | 24h | 0.293675 | 422448.80855333 | 0.2 | 0.6 | 0.4 |
| 0.5 | BABY | execution_watch | both_maker | 8h | 0.42155 | 18182.63949194 | 0 | 0.2 | 0.2 |
| 0.5 | JTO | drop_current | both_maker | blocked | -0.059225 | 54543.56750198 | 0 | 0.8 | 0.2 |
| 1 | ZEC | paper_24h_candidate | okx_cross_hl_maker | 24h | 0.74465 | 106210.05564167 | 0 | 0.6 | 0.2 |
| 1 | BABY | execution_watch | both_maker | 24h | 1.764675 | 18182.63949194 | 0 | 0.2 | 0.2 |
| 1 | BTC | drop_current | both_maker | blocked | -0.206325 | 422448.80855333 | 0.2 | 0.6 | 0.4 |
| 1 | JTO | drop_current | both_maker | blocked | -0.559225 | 54543.56750198 | 0 | 0.8 | 0.2 |

## Interpretation

Fee sensitivity separates fee-robust candidates from candidates that only look alive under maker-only or very-low-fee assumptions. A candidate that falls from `paper_*` to `execution_watch` still has raw edge, but the current execution proxy does not support promotion.
