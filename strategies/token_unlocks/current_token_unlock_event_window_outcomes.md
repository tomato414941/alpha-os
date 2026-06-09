# Current Paper Ticket Outcomes

This checks opened paper tickets against the latest available public marks. It is not a fill report and not a live trading PnL report.

| ticket | status | decision | asset | venue | entry | current | dir bps | outcome | missing evidence | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| unlock-event-hype-paper-short | ready | paper_short | HYPE | HL | 63.323000000000 | 62.290000000000 | 163.13187941 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| unlock-event-zro-paper-short | ready | paper_short | ZRO | HL | 0.848880000000 | 0.827490000000 | 251.97907831 | paper_mark_win | fill, funding, stop, and adverse excursion still missing | record fill, funding, stop, and adverse-excursion assumptions before promotion |
| unlock-event-me-paper-long | ready | paper_long | ME | HL | 0.061590000000 | 0.060500000000 | -176.97678195 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| unlock-event-eigen-paper-short | ready | paper_short | EIGEN | HL | 0.180600000000 | 0.181600000000 | -55.37098560 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| unlock-event-kaito-paper-long | pending | paper_long | KAITO | HL | 0.424200000000 | 0.424200000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |

## Summary

- ready: 4
- wins: 2
- losses: 2
- flat: 0
- observations: 0
- pending: 1
- best ready mark: unlock-event-zro-paper-short ZRO 251.97907831bps paper_mark_win
- worst ready mark: unlock-event-me-paper-long ME -176.97678195bps paper_mark_loss
