# Current Repeat Execution Gate

This joins repeat-priority labels to current HL/OKX public execution context. It is a paper-check queue, not a trade instruction.

| asset | source | venue | direction | labels | hit15 | mean15 bps | spread bps | depth 10bps USD | rough net15 bps | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SUI | on_chain_flow | HL | long | 6 | 1.000 | 63.35 | 0.13 | 75354 | 55.21 | small_repeat_paper_check | paper-check HL SUI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| SUI | on_chain_flow | OKX | long | 6 | 1.000 | 63.35 | 1.34 | 104996 | 54.01 | small_repeat_paper_check | paper-check OKX SUI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| CHIP | exchange_catalyst | OKX | long | 3 | 1.000 | 39.24 | 3.24 | 6075 | 28.00 | small_repeat_paper_check | paper-check OKX CHIP/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| JTO | liquidation | HL | short | 2 | 1.000 | 165.39 |  |  |  | missing_venue_context | collect HL execution context for JTO/liquidation |
| JTO | liquidation | OKX | short | 2 | 1.000 | 165.39 |  |  |  | missing_venue_context | collect OKX execution context for JTO/liquidation |
| CHIP | exchange_catalyst | HL | long | 3 | 1.000 | 39.24 |  |  |  | missing_venue_context | collect HL execution context for CHIP/exchange_catalyst |

## Interpretation

`small_repeat_paper_check` means repeated 15m labels are still positive after a rough spread plus taker-cost haircut and visible 10bps depth is not obviously blocking a 1k paper check. It still needs 1h confirmation, real fills, funding PnL, stop behavior, and adverse-selection checks.
