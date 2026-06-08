# Current Paper Ticket Outcomes

This checks opened paper tickets against the latest available public marks. It is not a fill report and not a live trading PnL report.

| ticket | status | decision | asset | venue | entry | current | dir bps | outcome | missing evidence | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| repeat-paper-26-sui-liquidation-intensity | pending | paper_long | SUI |  | 0.750000000000 | 0.750600000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-paper-27-pepe-liquidation-intensity | pending | paper_long | PEPE |  | 0.000002783000 | 0.000000000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-paper-01-sui-repeat-execution | pending | paper_long | SUI |  | 0.750600000000 | 0.750600000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-paper-19-sui-microstructure-flow | pending | paper_long | SUI |  | 0.750600000000 | 0.750600000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-paper-02-sui-repeat-execution | pending | paper_long | SUI |  | 0.750000000000 | 0.750600000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-paper-09-sol-volume-dislocation | ready | paper_long | SOL |  | 65.893000000000 | 65.880000000000 | -1.97289545 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| repeat-paper-05-mega-microstructure-flow | ready | paper_long | MEGA |  | 0.051157000000 | 0.051352000000 | 38.11795062 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |

## Summary

- ready: 2
- wins: 1
- losses: 1
- flat: 0
- pending: 5
- best ready mark: repeat-paper-05-mega-microstructure-flow MEGA 38.11795062bps paper_mark_win
- worst ready mark: repeat-paper-09-sol-volume-dislocation SOL -1.97289545bps paper_mark_loss
