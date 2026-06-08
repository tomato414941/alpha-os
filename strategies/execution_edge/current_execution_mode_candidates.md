# Current Execution Mode Candidates

This turns paper-ticket fill-risk checks into execution-mode candidates. It is not a live order router and does not assume maker fills.

| ticket | asset | mode | action | score | current net | mode net | spread | usage | suggested size | reason |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| paper-05-hype-volume-dislocation | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 554.0983 | 514.0210 | 520.0983 | 0.1546 | 0.0026 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-17-hype-policy-expansion | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 554.0983 | 514.0210 | 520.0983 | 0.1546 | 0.0010 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-29-hype-policy-expansion | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 554.0983 | 514.0210 | 520.0983 | 0.1546 | 0.0010 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-05-hype-volume-dislocation | HYPE | taker_small | repeat_taker_probe | 550.0210 | 514.0210 | 514.0210 | 0.1546 | 0.0026 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-17-hype-policy-expansion | HYPE | taker_small | repeat_taker_probe | 550.0210 | 514.0210 | 514.0210 | 0.1546 | 0.0010 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-29-hype-policy-expansion | HYPE | taker_small | repeat_taker_probe | 550.0210 | 514.0210 | 514.0210 | 0.1546 | 0.0010 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-hype-hype-volume-price-dislocation | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 532.4064 | 492.3291 | 498.4064 | 0.1546 | 0.0026 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-hype-hype-attention-price-context | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 532.4064 | 492.3291 | 498.4064 | 0.1546 | 0.0010 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-hype-hype-microstructure-flow-paper-probe | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 532.4064 | 492.3291 | 498.4064 | 0.1546 | 0.0010 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-hype-hype-microstructure-flow-probe | HYPE | maker_or_low_fee_small | compare_taker_vs_low_fee | 532.4064 | 492.3291 | 498.4064 | 0.1546 | 0.0010 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-hype-hype-volume-price-dislocation | HYPE | taker_small | repeat_taker_probe | 528.3291 | 492.3291 | 492.3291 | 0.1546 | 0.0026 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-hype-hype-attention-price-context | HYPE | taker_small | repeat_taker_probe | 528.3291 | 492.3291 | 492.3291 | 0.1546 | 0.0010 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-hype-hype-microstructure-flow-paper-probe | HYPE | taker_small | repeat_taker_probe | 528.3291 | 492.3291 | 492.3291 | 0.1546 | 0.0010 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| lane-hype-hype-microstructure-flow-probe | HYPE | taker_small | repeat_taker_probe | 528.3291 | 492.3291 | 492.3291 | 0.1546 | 0.0010 | 100 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-22-bera-microstructure-flow | BERA | maker_or_low_fee_small | compare_taker_vs_low_fee | 468.9161 | 427.9558 | 434.9161 | 1.9206 | 0.0251 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-22-bera-microstructure-flow | BERA | taker_small | repeat_taker_probe | 463.9558 | 427.9558 | 427.9558 | 1.9206 | 0.0251 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-21-chip-repeat-execution | CHIP | reduced_size_taker | retry_with_depth_capped_size | 370.1728 | 346.4798 | 346.4798 | 3.0698 | 0.1863 | 268.41 | paper edge survives directionally but current size consumes too much visible depth |
| repeat-paper-21-chip-microstructure-flow | CHIP | maker_or_low_fee_small | compare_taker_vs_low_fee | 370.0018 | 330.5610 | 339.5610 | 6.4489 | 0.0204 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-21-chip-microstructure-flow | CHIP | taker_small | repeat_taker_probe | 363.0018 | 330.5610 | 330.5610 | 6.4489 | 0.0204 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| repeat-paper-20-mon-microstructure-flow | MON | maker_or_low_fee_small | compare_taker_vs_low_fee | 273.1232 | 232.1932 | 239.3100 | 2.2336 | 0.0201 | 100.00 | low-fee or maker-like execution may improve the already surviving paper edge |
| repeat-paper-20-mon-microstructure-flow | MON | taker_small | repeat_taker_probe | 268.0064 | 232.1932 | 232.1932 | 2.2336 | 0.0201 | 100.00 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-09-fartcoin-volume-dislocation | FARTCOIN | maker_or_low_fee_small | compare_taker_vs_low_fee | 228.5608 | 188.4743 | 196.9968 | 5.0450 | 0.0219 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-09-fartcoin-volume-dislocation | FARTCOIN | taker_small | repeat_taker_probe | 222.0383 | 188.4743 | 188.4743 | 5.0450 | 0.0219 | 250 | paper edge survives rough taker spread, fee, funding, and depth checks |
| paper-11-pump-volume-dislocation | PUMP | maker_or_low_fee_small | compare_taker_vs_low_fee | 200.1944 | 162.8148 | 171.8148 | 6.1633 | 0.0139 | 250 | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-44-zec-attention-event | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 195.6740 | 157.8843 | 164.8996 | 2.0305 | 0.0000 |  | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-46-zec-sector-rotation | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 195.6740 | 157.8843 | 164.8996 | 2.0305 | 0.0000 |  | low-fee or maker-like execution may improve the already surviving paper edge |
| paper-23-zec-wallet-entity-flow | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 195.6740 | 157.8843 | 164.8996 | 2.0305 | 0.0000 |  | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-zec-zec-attention-price-context | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 195.6740 | 157.8843 | 164.8996 | 2.0305 | 0.0014 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-zec-zec-institutional-flow-news-event | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 195.6740 | 157.8843 | 164.8996 | 2.0305 | 0.0014 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |
| lane-zec-zec-narrative-event-news-event | ZEC | maker_or_low_fee_small | compare_taker_vs_low_fee | 195.6740 | 157.8843 | 164.8996 | 2.0305 | 0.0014 | 100 | low-fee or maker-like execution may improve the already surviving paper edge |

## Interpretation

Execution can be an alpha source only when order mode, size, fee tier, visible depth, fill probability, queue position, and adverse selection are made explicit. Rows here identify where a paper edge survives or fails under simple execution choices.
