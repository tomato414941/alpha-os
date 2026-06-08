# Current BTC ETF Funding Paper Ticket

This compares current venues for the active BTC ETF-flow/funding paper watch. It is not a live trade instruction.

| venue | instrument | side | ann funding | carry side | volume USD | OI USD | spread/impact | basis/premium | score | status | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| OKX | BTC-USDT-SWAP | short | -0.05241027 | long_perp_receives_funding | 9125792364 | 1814074265 | 0.00000158 | -0.00065179 | 4.946780 | reject | short side does not receive funding |
| Hyperliquid | BTC-USD perpetual | short | 0.10950000 | short_perp_receives_funding | 3352209749 | 1919199226 | 0.00018810 | -0.00047004 | 3.442430 | paper_venue_candidate | short carry, liquidity, and visible friction are acceptable for paper watch |

## Caveat

A paper ticket still needs account fee tier, maker/taker behavior, margin mode, stop execution, mark/index basis, and funding timestamp checks. The current score only ranks visible public venue context.
