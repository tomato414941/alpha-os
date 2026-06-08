# Current Repeat Execution Gate

This joins repeat-priority labels to current HL/OKX public execution context. It is a paper-check queue, not a trade instruction.

| asset | source | venue | labels | hit15 | mean15 bps | spread bps | depth 10bps USD | rough net15 bps | gate | next step |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MEGA | exchange_catalyst | OKX | 2 | 1.000 | 105.12 | 2.01 | 5398 | 95.12 | small_repeat_paper_check | paper-check OKX MEGA/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| HYPE | on_chain_flow | HL | 4 | 1.000 | 100.11 | 0.32 | 113501 | 91.79 | small_repeat_paper_check | paper-check HL HYPE/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| HYPE | on_chain_flow | OKX | 4 | 1.000 | 100.11 | 1.60 | 311174 | 90.51 | small_repeat_paper_check | paper-check OKX HYPE/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | exchange_catalyst | OKX | 4 | 1.000 | 96.81 | 4.77 | 15209 | 84.04 | small_repeat_paper_check | paper-check OKX NEAR/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | on_chain_flow | OKX | 4 | 1.000 | 96.81 | 4.77 | 15209 | 84.04 | small_repeat_paper_check | paper-check OKX NEAR/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | exchange_catalyst | HL | 4 | 1.000 | 96.81 | 5.24 | 11665 | 83.56 | small_repeat_paper_check | paper-check HL NEAR/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | on_chain_flow | HL | 4 | 1.000 | 96.81 | 5.24 | 11665 | 83.56 | small_repeat_paper_check | paper-check HL NEAR/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| WLD | l2_imbalance | OKX | 3 | 1.000 | 68.63 | 2.11 | 33101 | 58.52 | small_repeat_paper_check | paper-check OKX WLD/l2_imbalance with 1h label, fill, funding, stop, and adverse-excursion logs |
| WLD | l2_imbalance | HL | 3 | 1.000 | 68.63 | 3.59 | 10352 | 57.04 | small_repeat_paper_check | paper-check HL WLD/l2_imbalance with 1h label, fill, funding, stop, and adverse-excursion logs |
| SUI | on_chain_flow | OKX | 4 | 1.000 | 63.46 | 1.35 | 94796 | 54.11 | small_repeat_paper_check | paper-check OKX SUI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| SUI | on_chain_flow | HL | 4 | 1.000 | 63.46 | 1.61 | 62554 | 53.84 | small_repeat_paper_check | paper-check HL SUI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| CHIP | exchange_catalyst | OKX | 2 | 1.000 | 47.56 | 3.26 | 6491 | 36.29 | small_repeat_paper_check | paper-check OKX CHIP/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| JTO | liquidation | HL | 2 | 1.000 | 165.39 |  |  |  | missing_venue_context | collect HL execution context for JTO/liquidation |
| JTO | liquidation | OKX | 2 | 1.000 | 165.39 |  |  |  | missing_venue_context | collect OKX execution context for JTO/liquidation |
| MEGA | exchange_catalyst | HL | 2 | 1.000 | 105.12 |  |  |  | missing_venue_context | collect HL execution context for MEGA/exchange_catalyst |
| CHIP | exchange_catalyst | HL | 2 | 1.000 | 47.56 |  |  |  | missing_venue_context | collect HL execution context for CHIP/exchange_catalyst |

## Interpretation

`small_repeat_paper_check` means repeated 15m labels are still positive after a rough spread plus taker-cost haircut and visible 10bps depth is not obviously blocking a 1k paper check. It still needs 1h confirmation, real fills, funding PnL, stop behavior, and adverse-selection checks.
