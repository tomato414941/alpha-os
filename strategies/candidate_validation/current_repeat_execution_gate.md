# Current Repeat Execution Gate

This joins repeat-priority labels to current HL/OKX public execution context. It is a paper-check queue, not a trade instruction.

| asset | source | venue | direction | labels | hit15 | mean15 bps | spread bps | depth 10bps USD | rough net15 bps | gate | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MEGA | exchange_catalyst | OKX | long | 2 | 1.000 | 105.12 | 2.03 | 8439 | 95.10 | small_repeat_paper_check | paper-check OKX MEGA/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| HYPE | on_chain_flow | HL | long | 4 | 1.000 | 100.11 | 0.16 | 110716 | 91.96 | small_repeat_paper_check | paper-check HL HYPE/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| HYPE | on_chain_flow | OKX | long | 4 | 1.000 | 100.11 | 1.59 | 273559 | 90.53 | small_repeat_paper_check | paper-check OKX HYPE/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | exchange_catalyst | HL | long | 4 | 1.000 | 96.81 | 2.75 | 57413 | 86.05 | small_repeat_paper_check | paper-check HL NEAR/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | on_chain_flow | HL | long | 4 | 1.000 | 96.81 | 2.75 | 57413 | 86.05 | small_repeat_paper_check | paper-check HL NEAR/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | exchange_catalyst | OKX | long | 4 | 1.000 | 96.81 | 4.59 | 33699 | 84.22 | small_repeat_paper_check | paper-check OKX NEAR/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| NEAR | on_chain_flow | OKX | long | 4 | 1.000 | 96.81 | 4.59 | 33699 | 84.22 | small_repeat_paper_check | paper-check OKX NEAR/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| WLD | l2_imbalance | HL | long | 3 | 1.000 | 68.63 | 1.88 | 22759 | 58.75 | small_repeat_paper_check | paper-check HL WLD/l2_imbalance with 1h label, fill, funding, stop, and adverse-excursion logs |
| WLD | l2_imbalance | OKX | long | 3 | 1.000 | 68.63 | 2.09 | 42748 | 58.53 | small_repeat_paper_check | paper-check OKX WLD/l2_imbalance with 1h label, fill, funding, stop, and adverse-excursion logs |
| SUI | on_chain_flow | HL | long | 4 | 1.000 | 63.46 | 0.13 | 75354 | 55.32 | small_repeat_paper_check | paper-check HL SUI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| SUI | on_chain_flow | OKX | long | 4 | 1.000 | 63.46 | 1.34 | 104996 | 54.12 | small_repeat_paper_check | paper-check OKX SUI/on_chain_flow with 1h label, fill, funding, stop, and adverse-excursion logs |
| CHIP | exchange_catalyst | OKX | long | 2 | 1.000 | 47.56 | 3.24 | 6075 | 36.32 | small_repeat_paper_check | paper-check OKX CHIP/exchange_catalyst with 1h label, fill, funding, stop, and adverse-excursion logs |
| JTO | liquidation | HL | short | 2 | 1.000 | 165.39 |  |  |  | missing_venue_context | collect HL execution context for JTO/liquidation |
| JTO | liquidation | OKX | short | 2 | 1.000 | 165.39 |  |  |  | missing_venue_context | collect OKX execution context for JTO/liquidation |
| MEGA | exchange_catalyst | HL | long | 2 | 1.000 | 105.12 |  |  |  | missing_venue_context | collect HL execution context for MEGA/exchange_catalyst |
| CHIP | exchange_catalyst | HL | long | 2 | 1.000 | 47.56 |  |  |  | missing_venue_context | collect HL execution context for CHIP/exchange_catalyst |

## Interpretation

`small_repeat_paper_check` means repeated 15m labels are still positive after a rough spread plus taker-cost haircut and visible 10bps depth is not obviously blocking a 1k paper check. It still needs 1h confirmation, real fills, funding PnL, stop behavior, and adverse-selection checks.
