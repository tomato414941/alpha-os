# Current Paper Ticket Outcomes

This checks opened paper tickets against the latest available public marks. It is not a fill report and not a live trading PnL report.

| ticket | status | decision | asset | venue | entry | current | dir bps | outcome | missing evidence | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| paper-01-sui-repeat-execution | pending | paper_long | SUI | HL | 0.747670000000 | 0.757110000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-02-sui-repeat-execution | pending | paper_long | SUI | OKX | 0.747500000000 | 0.756600000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-03-inj-volume-dislocation | ready | paper_long | INJ | HL | 5.529600000000 | 5.520500000000 | -16.45688657 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| paper-04-hype-volume-dislocation | ready | paper_long | HYPE | HL | 63.047000000000 | 63.038000000000 | -1.42750646 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| paper-05-mega-microstructure-flow | pending | paper_long | MEGA |  | 0.049314000000 | 0.050639000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-06-hype-microstructure-flow | pending | paper_long | HYPE |  | 63.047000000000 | 63.038000000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-07-fartcoin-volume-dislocation | ready | paper_long | FARTCOIN | HL | 0.115150000000 | 0.114960000000 | -16.50021711 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| paper-08-xpl-volume-dislocation | ready | paper_long | XPL | HL | 0.070094000000 | 0.069580000000 | -73.33009958 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| paper-09-sol-volume-dislocation | ready | paper_long | SOL | HL | 65.737000000000 | 65.893000000000 | 23.73092779 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| paper-10-arbusdt-intraday-derivatives | pending | paper_short | ARBUSDT |  | 0.081465000000 | 0.082015000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-11-arbusdt-intraday-derivatives | pending | paper_short | ARBUSDT |  | 0.081465000000 | 0.082015000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-12-virtual-volume-dislocation | ready | paper_long | VIRTUAL | HL | 0.579990000000 | 0.578550000000 | -24.82801428 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| paper-13-eth-volume-dislocation | ready | paper_long | ETH | HL | 1669.100000000000 | 1666.800000000000 | -13.77988137 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| paper-14-event-event-probability | ready | paper_long | EVENT | prediction_market | 0.100000 | 0.100000 | 0.00000000 | paper_mark_flat | fill, funding, stop, and adverse excursion still missing | keep observing until the ticket has a non-flat mark move or stronger quote evidence |
| paper-15-link-volume-dislocation | ready | paper_long | LINK | HL | 7.899800000000 | 7.891400000000 | -10.63318059 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| paper-16-chip-repeat-execution | pending | paper_long | CHIP | OKX | 0.030860000000 | 0.030880000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-17-event-event-probability | ready | paper_long | EVENT | prediction_market | 0.290000 | 0.290000 | 0.00000000 | paper_mark_flat | fill, funding, stop, and adverse excursion still missing | keep observing until the ticket has a non-flat mark move or stronger quote evidence |
| paper-18-near-microstructure-flow | pending | paper_long | NEAR |  | 2.176400000000 | 2.128000000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-19-sui-microstructure-flow | pending | paper_long | SUI |  | 0.747670000000 | 0.757110000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| paper-20-mon-microstructure-flow | pending | paper_long | MON |  | 0.021816000000 | 0.021779000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |

## Summary

- ready: 10
- wins: 1
- losses: 7
- flat: 2
- pending: 10
- best ready mark: paper-09-sol-volume-dislocation SOL 23.73092779bps paper_mark_win
- worst ready mark: paper-08-xpl-volume-dislocation XPL -73.33009958bps paper_mark_loss
