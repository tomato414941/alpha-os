# Current Execution Mode Candidates

This turns paper-ticket fill-risk checks into execution-mode candidates. It is not a live order router and does not assume maker fills.

| ticket | asset | mode | action | score | current net | mode net | spread | usage | suggested size | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lane-near-near-microstructure-flow-paper-probe | NEAR | maker_or_low_fee_small | compare_taker_vs_low_fee | 185.4809 | 148.6992 | 155.6153 | 1.8321 | 0.0033 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-near-near-microstructure-flow-probe | NEAR | maker_or_low_fee_small | compare_taker_vs_low_fee | 185.4809 | 148.6992 | 155.6153 | 1.8321 | 0.0033 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-03-inj-volume-dislocation | INJ | context_required | refresh_execution_context | 181.8429 | 153.6753 | 161.6753 | 0.0000 | 0.0000 | 250 | paper edge cannot be evaluated because spread, funding, or visible depth is missing |
| lane-near-near-microstructure-flow-paper-probe | NEAR | taker_small | repeat_taker_probe | 180.5648 | 148.6992 | 148.6992 | 1.8321 | 0.0033 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-near-near-microstructure-flow-probe | NEAR | taker_small | repeat_taker_probe | 180.5648 | 148.6992 | 148.6992 | 1.8321 | 0.0033 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-lane-near-near-microstructure-flow-paper-probe | NEAR | maker_or_low_fee_small | compare_taker_vs_low_fee | 168.3500 | 133.1257 | 140.0417 | 1.8321 | 0.0033 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-lane-near-near-microstructure-flow-paper-probe | NEAR | taker_small | repeat_taker_probe | 163.4340 | 133.1257 | 133.1257 | 1.8321 | 0.0033 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-lane-near-near-microstructure-flow-probe | NEAR | maker_or_low_fee_small | compare_taker_vs_low_fee | 143.5259 | 110.5583 | 117.4744 | 1.8321 | 0.0033 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-lane-near-near-microstructure-flow-probe | NEAR | taker_small | repeat_taker_probe | 138.6099 | 110.5583 | 110.5583 | 1.8321 | 0.0033 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-sol-sol-volume-price-dislocation | SOL | maker_or_low_fee_small | compare_taker_vs_low_fee | 102.0307 | 73.7830 | 79.8585 | 0.1510 | 0.0006 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-sol-solana-stablecoin-migration | SOL | maker_or_low_fee_small | compare_taker_vs_low_fee | 102.0307 | 73.7830 | 79.8585 | 0.1510 | 0.0003 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-sol-sol-attention-price-context | SOL | maker_or_low_fee_small | compare_taker_vs_low_fee | 102.0307 | 73.7830 | 79.8585 | 0.1510 | 0.0003 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-sol-sol-volume-price-dislocation | SOL | taker_small | repeat_taker_probe | 97.9552 | 73.7830 | 73.7830 | 0.1510 | 0.0006 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-sol-solana-stablecoin-migration | SOL | taker_small | repeat_taker_probe | 97.9552 | 73.7830 | 73.7830 | 0.1510 | 0.0003 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-sol-sol-attention-price-context | SOL | taker_small | repeat_taker_probe | 97.9552 | 73.7830 | 73.7830 | 0.1510 | 0.0003 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-01-sui-repeat-execution | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 84.4780 | 57.1527 | 63.8173 | 1.3293 | 0.0124 | 1000 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-02-sui-repeat-execution | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 83.7011 | 56.4814 | 63.1459 | 1.3290 | 0.0099 | 1000 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-01-sui-repeat-execution | SUI | taker_small | repeat_taker_probe | 79.8133 | 57.1527 | 57.1527 | 1.3293 | 0.0124 | 1000 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-02-sui-repeat-execution | SUI | taker_small | repeat_taker_probe | 79.0366 | 56.4814 | 56.4814 | 1.3290 | 0.0099 | 1000 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-lane-sol-sol-volume-price-dislocation | SOL | maker_or_low_fee_small | compare_taker_vs_low_fee | 73.5821 | 47.9207 | 53.9962 | 0.1510 | 0.0006 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-09-sol-volume-dislocation | SOL | maker_or_low_fee_small | compare_taker_vs_low_fee | 71.3998 | 45.9368 | 52.0123 | 0.1510 | 0.0006 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-lane-sol-sol-volume-price-dislocation | SOL | taker_small | repeat_taker_probe | 69.5066 | 47.9207 | 47.9207 | 0.1510 | 0.0006 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-09-sol-volume-dislocation | SOL | taker_small | repeat_taker_probe | 67.3243 | 45.9368 | 45.9368 | 0.1510 | 0.0006 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-02-sui-repeat-execution | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 50.0770 | 25.8791 | 32.5437 | 1.3293 | 0.0124 | 1000 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-26-sui-liquidation-intensity | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 50.0770 | 25.8791 | 32.5437 | 1.3293 | 0.0012 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-02-sui-repeat-execution | SUI | taker_small | repeat_taker_probe | 45.4124 | 25.8791 | 25.8791 | 1.3293 | 0.0124 | 1000 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-26-sui-liquidation-intensity | SUI | taker_small | repeat_taker_probe | 45.4124 | 25.8791 | 25.8791 | 1.3293 | 0.0012 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-01-sui-repeat-execution | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 41.2530 | 17.8572 | 24.5218 | 1.3293 | 0.0124 | 1000 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-19-sui-microstructure-flow | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 41.2530 | 17.8572 | 24.5218 | 1.3293 | 0.0012 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-01-sui-repeat-execution | SUI | taker_small | repeat_taker_probe | 36.5884 | 17.8572 | 17.8572 | 1.3293 | 0.0124 | 1000 | paper edge survives rough taker spread, fee, funding, and depth checks |

## Interpretation

Execution can be an alpha source only when order mode, size, fee tier, visible depth, fill probability, queue position, and adverse selection are made explicit. Rows here identify where a paper edge survives or fails under simple execution choices.
