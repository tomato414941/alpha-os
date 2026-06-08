# Current Paper Ticket Outcomes

This checks opened paper tickets against the latest available public marks. It is not a fill report and not a live trading PnL report.

| ticket | status | decision | asset | venue | entry | current | dir bps | outcome | missing evidence | next step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| portable-micro-hype-1h | pending | paper_short | HYPE |  | 64.919000000000 | 63.549000000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |
| portable-micro-eth-15m | ready | paper_long | ETH |  | 1703.450000000000 | 1687.500000000000 | -93.63350847 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| portable-micro-btc-15m | ready | paper_long | BTC |  | 64181.000000000000 | 63481.000000000000 | -109.06654617 | paper_mark_loss | fill, funding, stop, and adverse excursion still missing | keep or reject based on repeated labels and failure regime |
| portable-micro-sol-1h | pending | paper_short | SOL |  | 67.741500000000 | 67.469000000000 |  | pending | checkpoint has not matured | wait for the first checkpoint and refresh marks |

## Summary

- ready: 2
- wins: 0
- losses: 2
- flat: 0
- observations: 0
- pending: 2
- best ready mark: portable-micro-eth-15m ETH -93.63350847bps paper_mark_loss
- worst ready mark: portable-micro-btc-15m BTC -109.06654617bps paper_mark_loss
