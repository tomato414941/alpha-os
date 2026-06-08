# Current Repeat Execution Gate

This joins repeat-priority labels to current HL/OKX public execution context. It is a paper-check queue, not a trade instruction.

| asset | source | venue | labels | hit15 | mean15 bps | spread bps | depth 10bps USD | rough net15 bps | gate | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MEGA | exchange_catalyst | OKX | 2 | 1.000 | 106.13 | 2.01 | 5398 | 96.12 | small_repeat_paper_check | paper-check OKX MEGA/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | exchange_catalyst | OKX | 4 | 1.000 | 67.54 | 4.77 | 15209 | 54.78 | small_repeat_paper_check | paper-check OKX NEAR/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | on_chain_flow | OKX | 4 | 1.000 | 67.54 | 4.77 | 15209 | 54.78 | small_repeat_paper_check | paper-check OKX NEAR/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | exchange_catalyst | HL | 4 | 1.000 | 67.54 | 5.24 | 11665 | 54.30 | small_repeat_paper_check | paper-check HL NEAR/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | on_chain_flow | HL | 4 | 1.000 | 67.54 | 5.24 | 11665 | 54.30 | small_repeat_paper_check | paper-check HL NEAR/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| SUI | on_chain_flow | OKX | 4 | 1.000 | 57.35 | 1.35 | 94796 | 48.01 | small_repeat_paper_check | paper-check OKX SUI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| SUI | on_chain_flow | HL | 4 | 1.000 | 57.35 | 1.61 | 62554 | 47.74 | small_repeat_paper_check | paper-check HL SUI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| CHIP | exchange_catalyst | OKX | 2 | 1.000 | 42.69 | 3.26 | 6491 | 31.43 | small_repeat_paper_check | paper-check OKX CHIP/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| SEI | on_chain_flow | OKX | 2 | 1.000 | 30.75 | 2.04 | 7852 | 20.71 | small_repeat_paper_check | paper-check OKX SEI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| ARB | on_chain_flow | OKX | 4 | 0.750 | 11.69 | 1.22 | 14952 | 2.47 | small_repeat_paper_check | paper-check OKX ARB/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| ARB | on_chain_flow | HL | 4 | 0.750 | 11.69 | 3.68 | 21859 | 0.02 | small_repeat_paper_check | paper-check HL ARB/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| JTO | liquidation | HL | 2 | 1.000 | 165.39 |  |  |  | missing_venue_context | collect HL execution context for JTO/liquidation |
| JTO | liquidation | OKX | 2 | 1.000 | 165.39 |  |  |  | missing_venue_context | collect OKX execution context for JTO/liquidation |
| MEGA | exchange_catalyst | HL | 2 | 1.000 | 106.13 |  |  |  | missing_venue_context | collect HL execution context for MEGA/exchange_catalyst |
| CHIP | exchange_catalyst | HL | 2 | 1.000 | 42.69 |  |  |  | missing_venue_context | collect HL execution context for CHIP/exchange_catalyst |
| SEI | on_chain_flow | HL | 2 | 1.000 | 30.75 |  |  |  | missing_venue_context | collect HL execution context for SEI/on_chain_flow |

## Interpretation

`small_repeat_paper_check` means repeated 15m labels are still positive after a rough spread plus taker-cost haircut and visible 10bps depth is not obviously blocking a 1k paper check. It still needs 1h confirmation, real fills, funding PnL, stop behavior, and adverse-selection checks.
