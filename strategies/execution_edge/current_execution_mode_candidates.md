# Current Execution Mode Candidates

This turns paper-ticket fill-risk checks into execution-mode candidates. It is not a live order router and does not assume maker fills.

| ticket | asset | mode | action | score | current net | mode net | spread | usage | suggested size | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| paper-01-hype-policy-expansion | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 554.0983 | 514.0210 | 520.0983 | 0.1546 | 0.0010 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-05-hype-volume-dislocation | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 554.0983 | 514.0210 | 520.0983 | 0.1546 | 0.0026 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-17-hype-policy-expansion | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 554.0983 | 514.0210 | 520.0983 | 0.1546 | 0.0010 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-29-hype-policy-expansion | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 554.0983 | 514.0210 | 520.0983 | 0.1546 | 0.0010 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-01-hype-policy-expansion | HYPE | taker_small | repeat_taker_probe | 550.0210 | 514.0210 | 514.0210 | 0.1546 | 0.0010 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-05-hype-volume-dislocation | HYPE | taker_small | repeat_taker_probe | 550.0210 | 514.0210 | 514.0210 | 0.1546 | 0.0026 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-17-hype-policy-expansion | HYPE | taker_small | repeat_taker_probe | 550.0210 | 514.0210 | 514.0210 | 0.1546 | 0.0010 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-29-hype-policy-expansion | HYPE | taker_small | repeat_taker_probe | 550.0210 | 514.0210 | 514.0210 | 0.1546 | 0.0010 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-22-bera-microstructure-flow | BERA | maker_or_low_fee_small | compare_taker_vs_low_fee | 468.9161 | 427.9558 | 434.9161 | 1.9206 | 0.0251 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-22-bera-microstructure-flow | BERA | taker_small | repeat_taker_probe | 463.9558 | 427.9558 | 427.9558 | 1.9206 | 0.0251 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-21-chip-repeat-execution | CHIP | reduced_size_taker | retry_with_depth_capped_size | 370.1728 | 346.4798 | 346.4798 | 3.0698 | 0.1863 | 268.41 | paper edge survives directionally but current size consumes too much visible depth |
| repeat-paper-21-chip-microstructure-flow | CHIP | maker_or_low_fee_small | compare_taker_vs_low_fee | 370.0018 | 330.5610 | 339.5610 | 6.4489 | 0.0204 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-21-chip-microstructure-flow | CHIP | taker_small | repeat_taker_probe | 363.0018 | 330.5610 | 330.5610 | 6.4489 | 0.0204 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-20-mon-microstructure-flow | MON | maker_or_low_fee_small | compare_taker_vs_low_fee | 273.1232 | 232.1932 | 239.3100 | 2.2336 | 0.0201 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-20-mon-microstructure-flow | MON | taker_small | repeat_taker_probe | 268.0064 | 232.1932 | 232.1932 | 2.2336 | 0.0201 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-09-fartcoin-volume-dislocation | FARTCOIN | maker_or_low_fee_small | compare_taker_vs_low_fee | 229.5267 | 189.2173 | 197.3683 | 4.3020 | 0.0130 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-09-fartcoin-volume-dislocation | FARTCOIN | taker_small | repeat_taker_probe | 223.3757 | 189.2173 | 189.2173 | 4.3020 | 0.0130 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-11-pump-volume-dislocation | PUMP | maker_or_low_fee_small | compare_taker_vs_low_fee | 199.9816 | 162.7074 | 171.7074 | 6.2952 | 0.0108 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-23-zec-wallet-entity-flow | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 197.9813 | 161.0635 | 167.1824 | 0.2378 | 0.0000 |  | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-46-zec-sector-rotation | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 197.9813 | 161.0635 | 167.1824 | 0.2378 | 0.0000 |  | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-23-zec-wallet-entity-flow | ZEC | taker_small | repeat_taker_probe | 193.8624 | 161.0635 | 161.0635 | 0.2378 | 0.0000 |  | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-46-zec-sector-rotation | ZEC | taker_small | repeat_taker_probe | 193.8624 | 161.0635 | 161.0635 | 0.2378 | 0.0000 |  | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-11-pump-volume-dislocation | PUMP | taker_small | repeat_taker_probe | 192.9816 | 162.7074 | 162.7074 | 6.2952 | 0.0108 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-near-near-microstructure-flow-paper-probe | NEAR | maker_or_low_fee_small | compare_taker_vs_low_fee | 185.4809 | 148.6992 | 155.6153 | 1.8321 | 0.0033 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-near-near-microstructure-flow-probe | NEAR | maker_or_low_fee_small | compare_taker_vs_low_fee | 185.4809 | 148.6992 | 155.6153 | 1.8321 | 0.0033 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-02-sui-repeat-execution | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 181.0431 | 145.0911 | 151.6164 | 1.0505 | 0.0193 | 1000 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-26-sui-liquidation-intensity | SUI | maker_or_low_fee_small | compare_taker_vs_low_fee | 181.0431 | 145.0911 | 151.6164 | 1.0505 | 0.0019 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-near-near-microstructure-flow-paper-probe | NEAR | taker_small | repeat_taker_probe | 180.5648 | 148.6992 | 148.6992 | 1.8321 | 0.0033 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-near-near-microstructure-flow-probe | NEAR | taker_small | repeat_taker_probe | 180.5648 | 148.6992 | 148.6992 | 1.8321 | 0.0033 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-02-sui-repeat-execution | SUI | taker_small | repeat_taker_probe | 176.5178 | 145.0911 | 145.0911 | 1.0505 | 0.0193 | 1000 | paper edge survives rough taker spread, fee, funding, and depth checks |

## Interpretation

Execution can be an alpha source only when order mode, size, fee tier, visible depth, fill probability, queue position, and adverse selection are made explicit. Rows here identify where a paper edge survives or fails under simple execution choices.
