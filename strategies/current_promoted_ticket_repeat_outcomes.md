# Current Paper Ticket Outcomes

This checks opened paper tickets against the latest available public marks. It is not a fill report and not a live trading PnL report.

| ticket | status | decision | asset | venue | entry | current | dir bps | outcome | missing evidence | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| repeat-paper-36-near-derivatives-positioning | pending | paper_long | NEAR |  | 2.183400000000 | 2.183400000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-paper-27-pepe-liquidation-intensity | ready | paper_long | PEPE |  | 0.000002783000 | 0.000000000000 |  | missing_current_mark | entry or current mark is invalid | fill missing current mark before judging the ticket |
| repeat-paper-09-sol-volume-dislocation | ready | paper_long | SOL |  | 65.893000000000 | 66.248000000000 | 53.87522195 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-paper-26-sui-liquidation-intensity | ready | paper_long | SUI |  | 0.750000000000 | 0.752650000000 | 35.33333333 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-paper-01-sui-repeat-execution | ready | paper_long | SUI |  | 0.750600000000 | 0.752650000000 | 27.31148415 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-paper-19-sui-microstructure-flow | ready | paper_long | SUI |  | 0.750600000000 | 0.752650000000 | 27.31148415 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-paper-02-sui-repeat-execution | ready | paper_long | SUI |  | 0.750000000000 | 0.752650000000 | 35.33333333 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-paper-18-near-microstructure-flow | pending | paper_long | NEAR |  | 2.183400000000 | 2.183400000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-paper-05-mega-microstructure-flow | ready | paper_long | MEGA |  | 0.051157000000 | 0.050216000000 | -183.94354634 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| repeat-paper-20-mon-microstructure-flow | ready | paper_long | MON |  | 0.021851000000 | 0.021781000000 | -32.03514713 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| repeat-paper-50-hype-token-unlock | ready | paper_short | HYPE |  | 61.425000000000 | 61.478000000000 | -8.62840863 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |

## Summary

- ready: 9
- wins: 5
- losses: 3
- flat: 0
- observations: 0
- pending: 2
- best ready mark: repeat-paper-09-sol-volume-dislocation SOL 53.87522195bps paper_mark_win
- worst ready mark: repeat-paper-05-mega-microstructure-flow MEGA -183.94354634bps paper_mark_loss
