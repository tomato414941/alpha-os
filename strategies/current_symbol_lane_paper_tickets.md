# Current Symbol Lane Paper Tickets

These tickets open separate paper observations for the top symbol's lanes. They deliberately do not collapse conflicting hypotheses into one trade.

| ticket | symbol | bias | opportunity | decision | size USD | checkpoints | entry | support | next step |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- | --- |
| lane-hype-hype-volume-price-dislocation | HYPE | long | hype_volume_price_dislocation | paper_long | 250 | 15m,1h | 63.038000000000 | paper_execution_gated | paper-probe HYPE lane at the gated size and log outcome |
| lane-hype-hype-microstructure-flow-paper-probe | HYPE | long | hype_microstructure_flow_paper_probe | paper_long | 100.00 | 15m,1h | 63.038000000000 | paper_execution_gated | paper-probe HYPE lane at the gated size and log outcome |
| lane-hype-hype-microstructure-flow-probe | HYPE | long | hype_microstructure_flow_probe | paper_long | 100 | 1h | 63.038000000000 | paper_1h_supported | rerun HYPE lane on a fresh window and add execution/fill evidence |
| lane-hype-hype-protocol-fee-actionability | HYPE | short | hype_protocol_fee_actionability | paper_short | 100 | 4h,12h | 63.038000000000 | paper_4h_supported | repeat HYPE lane on another 4h window and refresh execution evidence |
| lane-hype-hyperliquid-l1-stablecoin-migration | HYPE | neutral | hyperliquid_l1_stablecoin_migration | paper_observe | 100 | 4h,12h | 63.038000000000 | paper_4h_supported | repeat HYPE lane on another 4h window and refresh execution evidence |
| lane-hype-hype-attention-price-context | HYPE | long | hype_attention_price_context | paper_long | 100 | 4h,12h | 63.038000000000 | unlabeled | paper-label HYPE attention-price lag over 1h, 4h, 12h, and 24h |
| lane-hype-hype-protocol-fee-growth | HYPE | short | hype_protocol_fee_growth | paper_short | 100 | 15m,1h | 63.038000000000 | unlabeled | separate HYPE protocol growth thesis from unlock short pressure and label both windows |
| lane-hype-hype-unlock-actionability | HYPE | short | hype_unlock_actionability | paper_short | 100 | 15m,1h | 63.038000000000 | pending_label | label HYPE unlock event window before treating the supply event as a tradable alpha |
