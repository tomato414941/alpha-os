# Current Paper Ticket Outcomes

This checks opened paper tickets against the latest available public marks. It is not a fill report and not a live trading PnL report.

| ticket | status | decision | asset | venue | entry | current | dir bps | outcome | missing evidence | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| repeat-repeat-paper-22-bera-microstructure-flow | pending | paper_long | BERA |  | 0.260320000000 | 0.260320000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-repeat-paper-21-chip-microstructure-flow | pending | paper_long | CHIP |  | 0.032564000000 | 0.032564000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-repeat-paper-20-mon-microstructure-flow | pending | paper_long | MON |  | 0.022381000000 | 0.022381000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| repeat-repeat-paper-02-sui-repeat-execution | ready | paper_long | SUI |  | 0.752730000000 | 0.761570000000 | 117.43918802 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-repeat-paper-26-sui-liquidation-intensity | ready | paper_long | SUI |  | 0.752730000000 | 0.761570000000 | 117.43918802 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-repeat-paper-01-sui-repeat-execution | ready | paper_long | SUI |  | 0.752730000000 | 0.761570000000 | 117.43918802 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-repeat-paper-19-sui-microstructure-flow | ready | paper_long | SUI |  | 0.752730000000 | 0.761570000000 | 117.43918802 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-repeat-paper-09-sol-volume-dislocation | ready | paper_long | SOL |  | 66.215000000000 | 66.720000000000 | 76.26670694 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| repeat-repeat-paper-05-mega-microstructure-flow | ready | paper_long | MEGA |  | 0.051591000000 | 0.050205000000 | -268.65150898 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |

## Summary

- ready: 6
- wins: 5
- losses: 1
- flat: 0
- observations: 0
- pending: 3
- best ready mark: repeat-repeat-paper-02-sui-repeat-execution SUI 117.43918802bps paper_mark_win
- worst ready mark: repeat-repeat-paper-05-mega-microstructure-flow MEGA -268.65150898bps paper_mark_loss
