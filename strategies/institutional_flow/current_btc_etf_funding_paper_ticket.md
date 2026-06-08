# Current BTC ETF Funding Paper Ticket

This compares current venues for the active BTC ETF-flow/funding paper watch. It is not a live trade instruction.

| venue | instrument | side | ann funding | carry side | volume USD | OI USD | spread/impact | basis/premium | score | status | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| OKX | BTC-USDT-SWAP | short | -0.04578734 | long_perp_receives_funding | 9155704735 | 1813212307 | 0.00000159 | -0.00068246 | 4.953371 | reject | short side does not receive funding |
| Hyperliquid | BTC-USD perpetual | short | 0.10950000 | short_perp_receives_funding | 2979459076 | 1929259224 | 0.00005835 | -0.00037829 | 3.082746 | paper_venue_candidate | short carry, liquidity, and visible friction are acceptable for paper watch |

## Caveat

A paper ticket still needs account fee tier, maker/taker behavior, margin mode, stop execution, mark/index basis, and funding timestamp checks. The current score only ranks visible public venue context.
