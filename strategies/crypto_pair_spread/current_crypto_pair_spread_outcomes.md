# Current Crypto Pair Spread Outcomes

This checks pair-ratio paper labels against current public marks. It is not a fill report and not a deployable pair execution overlay.

| ticket | status | pair | decision | entry ratio | current ratio | dir bps | outcome | next step |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pair-spread-btc-hype-mean-reversion | ready | BTC/HYPE | paper_long | 1000.679616253892 | 1000.284584980237 | -3.94762986 | paper_mark_loss | keep as a negative pair-spread label before trying execution overlay logic |
| pair-spread-eth-hype-mean-reversion | ready | ETH/HYPE | paper_long | 26.751592356688 | 26.798418972332 | 17.50423490 | paper_mark_win | repeat the pair label with explicit both-leg costs, funding carry, and hedge execution notes |
| pair-spread-sol-hype-mean-reversion | ready | SOL/HYPE | paper_long | 1.059379494555 | 1.059968379447 | 5.55877186 | paper_mark_win | repeat the pair label with explicit both-leg costs, funding carry, and hedge execution notes |
| pair-spread-btc-sol-mean-reversion | ready | BTC/SOL | paper_long | 944.590320463090 | 943.692853840073 | -9.50112026 | paper_mark_loss | keep as a negative pair-spread label before trying execution overlay logic |
| pair-spread-btc-eth-mean-reversion | ready | BTC/ETH | paper_long | 37.406357083776 | 37.326253687316 | -21.41438052 | paper_mark_loss | keep as a negative pair-spread label before trying execution overlay logic |
| pair-spread-eth-sol-mean-reversion | ready | ETH/SOL | paper_long | 25.252133436773 | 25.282281520815 | 11.93882652 | paper_mark_win | repeat the pair label with explicit both-leg costs, funding carry, and hedge execution notes |
