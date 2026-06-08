# Current Execution Mode Candidates

This turns paper-ticket fill-risk checks into execution-mode candidates. It is not a live order router and does not assume maker fills.

| ticket | asset | mode | action | score | current net | mode net | spread | usage | suggested size | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| repeat-paper-22-bera-microstructure-flow | BERA | reduced_size_taker | retry_with_depth_capped_size | 681.3901 | 659.1107 | 659.1107 | 6.3810 | 0.1554 | 32.18 | paper edge survives directionally but current size consumes too much visible depth |
| lane-hype-hype-volume-price-dislocation | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 556.7227 | 516.6457 | 522.7227 | 0.1542 | 0.0016 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-hype-hype-attention-price-context | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 556.7227 | 516.6457 | 522.7227 | 0.1542 | 0.0006 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-hype-hype-microstructure-flow-paper-probe | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 556.7227 | 516.6457 | 522.7227 | 0.1542 | 0.0006 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-hype-hype-microstructure-flow-probe | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 556.7227 | 516.6457 | 522.7227 | 0.1542 | 0.0006 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-hype-hype-volume-price-dislocation | HYPE | taker_small | repeat_taker_probe | 552.6457 | 516.6457 | 516.6457 | 0.1542 | 0.0016 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-hype-hype-attention-price-context | HYPE | taker_small | repeat_taker_probe | 552.6457 | 516.6457 | 516.6457 | 0.1542 | 0.0006 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-hype-hype-microstructure-flow-paper-probe | HYPE | taker_small | repeat_taker_probe | 552.6457 | 516.6457 | 516.6457 | 0.1542 | 0.0006 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-hype-hype-microstructure-flow-probe | HYPE | taker_small | repeat_taker_probe | 552.6457 | 516.6457 | 516.6457 | 0.1542 | 0.0006 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-zec-zec-attention-price-context | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 386.9424 | 346.4231 | 354.2243 | 3.6024 | 0.0026 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-zec-zec-institutional-flow-news-event | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 386.9424 | 346.4231 | 354.2243 | 3.6024 | 0.0026 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-zec-zec-narrative-event-news-event | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 386.9424 | 346.4231 | 354.2243 | 3.6024 | 0.0026 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-21-chip-microstructure-flow | CHIP | maker_or_low_fee_small | compare_taker_vs_low_fee | 386.6418 | 346.9475 | 355.9475 | 6.1321 | 0.0266 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-zec-zec-attention-price-context | ZEC | taker_small | repeat_taker_probe | 381.1412 | 346.4231 | 346.4231 | 3.6024 | 0.0026 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-zec-zec-institutional-flow-news-event | ZEC | taker_small | repeat_taker_probe | 381.1412 | 346.4231 | 346.4231 | 3.6024 | 0.0026 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-zec-zec-narrative-event-news-event | ZEC | taker_small | repeat_taker_probe | 381.1412 | 346.4231 | 346.4231 | 3.6024 | 0.0026 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-21-chip-microstructure-flow | CHIP | taker_small | repeat_taker_probe | 379.6418 | 346.9475 | 346.9475 | 6.1321 | 0.0266 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-repeat-paper-21-chip-microstructure-flow | CHIP | maker_or_low_fee_small | compare_taker_vs_low_fee | 363.9524 | 327.8966 | 336.8966 | 10.6803 | 0.0265 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-repeat-paper-21-chip-microstructure-flow | CHIP | taker_small | repeat_taker_probe | 356.9524 | 327.8966 | 327.8966 | 10.6803 | 0.0265 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-repeat-paper-22-bera-microstructure-flow | BERA | maker_or_low_fee_small | compare_taker_vs_low_fee | 354.5788 | 313.8345 | 320.5788 | 1.4885 | 0.0345 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-repeat-paper-22-bera-microstructure-flow | BERA | taker_small | repeat_taker_probe | 349.8345 | 313.8345 | 313.8345 | 1.4885 | 0.0345 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-sol-sol-volume-price-dislocation | SOL | maker_or_low_fee_small | compare_taker_vs_low_fee | 303.2928 | 263.2187 | 269.2928 | 0.1481 | 0.0007 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-sol-solana-stablecoin-migration | SOL | maker_or_low_fee_small | compare_taker_vs_low_fee | 303.2928 | 263.2187 | 269.2928 | 0.1481 | 0.0003 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-sol-sol-attention-price-context | SOL | maker_or_low_fee_small | compare_taker_vs_low_fee | 303.2928 | 263.2187 | 269.2928 | 0.1481 | 0.0003 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-sol-sol-volume-price-dislocation | SOL | taker_small | repeat_taker_probe | 299.2187 | 263.2187 | 263.2187 | 0.1481 | 0.0007 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-sol-solana-stablecoin-migration | SOL | taker_small | repeat_taker_probe | 299.2187 | 263.2187 | 263.2187 | 0.1481 | 0.0003 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-sol-sol-attention-price-context | SOL | taker_small | repeat_taker_probe | 299.2187 | 263.2187 | 263.2187 | 0.1481 | 0.0003 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-02-sui-repeat-execution | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 298.4767 | 258.4118 | 264.4767 | 0.1299 | 0.0087 | 1000 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-26-sui-liquidation-intensity | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 298.4767 | 258.4118 | 264.4767 | 0.1299 | 0.0009 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-02-sui-repeat-execution | SUI | taker_small | repeat_taker_probe | 294.4118 | 258.4118 | 258.4118 | 0.1299 | 0.0087 | 1000 | paper edge survives rough taker spread, fee, funding, and depth checks |

## Interpretation

Execution can be an alpha source only when order mode, size, fee tier, visible depth, fill probability, queue position, and adverse selection are made explicit. Rows here identify where a paper edge survives or fails under simple execution choices.
