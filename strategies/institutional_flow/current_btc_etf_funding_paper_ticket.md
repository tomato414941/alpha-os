# Current BTC ETF Funding Paper Ticket

This compares current venues for the active BTC ETF-flow/funding paper watch. It is not a live trade instruction.

| venue | instrument | side | ann funding | carry side | volume USD | OI USD | spread/impact | basis/premium | score | status | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| OKX | BTC-USDT-SWAP | short | -0.09020081 | long_perp_receives_funding | 9540450775 | 1860918108 | 0.00000156 | -0.00048200 | 4.909161 | reject | short side does not receive funding |
| Hyperliquid | BTC-USD perpetual | short | 0.10599600 | short_perp_receives_funding | 3349751910 | 1960071144 | 0.00001575 | -0.00044082 | 3.453732 | paper_venue_candidate | short carry, liquidity, and visible friction are acceptable for paper watch |

## Caveat

A paper ticket still needs account fee tier, maker/taker behavior, margin mode, stop execution, mark/index basis, and funding timestamp checks. The current score only ranks visible public venue context.
