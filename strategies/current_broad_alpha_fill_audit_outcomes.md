# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-zec-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZEC |  | long | -25.8696 | -53.6634 | stop_triggered | 14 | do not promote ZEC long; fresh fill audit hit the stop |
| broad-fill-audit-zec-long-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | ZEC |  | long | 64.7810 | -86.8022 | stop_triggered | 59 | do not promote ZEC long; fresh fill audit hit the stop |
| broad-fill-audit-pol-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | POL |  | short | -72.0379 | -75.5499 | stop_triggered | 14 | do not promote POL short; fresh fill audit hit the stop |
| broad-fill-audit-pol-short-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | POL |  | short | -98.4421 | -104.5538 | stop_triggered | 59 | do not promote POL short; fresh fill audit hit the stop |
| broad-fill-audit-wld-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | WLD |  | short | 437.0225 | -374.0546 | stop_triggered | 14 | do not promote WLD short; fresh fill audit hit the stop |
| broad-fill-audit-wld-short-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | WLD |  | short | 626.8909 | -374.0546 | stop_triggered | 59 | do not promote WLD short; fresh fill audit hit the stop |
| broad-fill-audit-eth-long-50bps-stop | 15m | ready | paper_fill_audit_loss | ETH |  | long | -23.3686 | -46.7372 | stop_survived | 14 | do not promote ETH long; fresh fill audit failed at 15m |
| broad-fill-audit-eth-long-50bps-stop | 1h | pending | pending | ETH |  | long | -116.2587 | -116.8429 | stop_triggered | 57 | wait for 1h fill-audit checkpoint for ETH long |
| broad-fill-audit-inj-long-50bps-stop | 15m | ready | paper_fill_audit_win | INJ |  | long | 98.4974 | -35.6599 | stop_survived | 14 | compare INJ long fill audit against prior broad paper path before promotion |
| broad-fill-audit-inj-long-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | INJ |  | long | -21.8114 | -98.8437 | stop_triggered | 59 | do not promote INJ long; fresh fill audit hit the stop |
| broad-fill-audit-zro-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZRO |  | short | -54.5720 | -65.0470 | stop_triggered | 14 | do not promote ZRO short; fresh fill audit hit the stop |
| broad-fill-audit-zro-short-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | ZRO |  | short | -47.3071 | -276.4027 | stop_triggered | 59 | do not promote ZRO short; fresh fill audit hit the stop |
| broad-fill-audit-sui-long-50bps-stop | 15m | ready | paper_fill_audit_loss | SUI |  | long | -10.4194 | -46.4965 | stop_survived | 14 | do not promote SUI long; fresh fill audit failed at 15m |
| broad-fill-audit-sui-long-50bps-stop | 1h | pending | pending | SUI |  | long | -109.6640 | -122.6882 | stop_triggered | 57 | wait for 1h fill-audit checkpoint for SUI long |
| broad-fill-audit-fet-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | FET |  | long | -18.5727 | -86.3630 | stop_triggered | 14 | do not promote FET long; fresh fill audit hit the stop |
| broad-fill-audit-fet-long-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | FET |  | long | -172.7260 | -188.9771 | stop_triggered | 59 | do not promote FET long; fresh fill audit hit the stop |
| broad-fill-audit-hype-long-50bps-stop | 15m | ready | paper_fill_audit_win | HYPE |  | long | 7.3713 | -11.4490 | stop_survived | 14 | compare HYPE long fill audit against prior broad paper path before promotion |
| broad-fill-audit-hype-long-50bps-stop | 1h | ready | paper_fill_audit_win | HYPE |  | long | 84.6913 | -11.4490 | stop_survived | 59 | compare HYPE long fill audit against prior broad paper path before promotion |
| broad-fill-audit-sol-long-50bps-stop | 15m | ready | paper_fill_audit_loss | SOL |  | long | -26.4858 | -41.2001 | stop_survived | 14 | do not promote SOL long; fresh fill audit failed at 15m |
| broad-fill-audit-sol-long-50bps-stop | 1h | pending | pending | SOL |  | long | -135.5189 | -135.5189 | stop_triggered | 57 | wait for 1h fill-audit checkpoint for SOL long |
| broad-fill-audit-allo-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ALLO | OKX | long | 186.9018 | -75.1616 | stop_triggered | 14 | do not promote ALLO long; fresh fill audit hit the stop |
| broad-fill-audit-allo-long-50bps-stop | 1h | pending | pending | ALLO | OKX | long | 179.8868 | -75.1616 | stop_triggered | 57 | wait for 1h fill-audit checkpoint for ALLO long |
| broad-fill-audit-apt-long-50bps-stop | 15m | ready | paper_fill_audit_loss | APT |  | long | -8.7642 | -45.2819 | stop_survived | 14 | do not promote APT long; fresh fill audit failed at 15m |
| broad-fill-audit-apt-long-50bps-stop | 1h | pending | pending | APT |  | long | -128.5422 | -128.5422 | stop_triggered | 57 | wait for 1h fill-audit checkpoint for APT long |
| broad-fill-audit-btc-long-50bps-stop | 15m | ready | paper_fill_audit_loss | BTC |  | long | -10.2031 | -19.3074 | stop_survived | 14 | do not promote BTC long; fresh fill audit failed at 15m |
| broad-fill-audit-btc-long-50bps-stop | 1h | pending | pending | BTC |  | long | -63.5733 | -63.5733 | stop_triggered | 57 | wait for 1h fill-audit checkpoint for BTC long |
| broad-fill-audit-chip-long-50bps-stop | 15m | ready | paper_fill_audit_win | CHIP |  | long | 97.6187 | -11.8326 | stop_survived | 14 | compare CHIP long fill audit against prior broad paper path before promotion |
| broad-fill-audit-chip-long-50bps-stop | 1h | ready | paper_fill_audit_win | CHIP |  | long | 309.4217 | -11.8326 | stop_survived | 59 | compare CHIP long fill audit against prior broad paper path before promotion |
| broad-fill-audit-wld-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | WLD |  | long | -418.7233 | -664.2264 | stop_triggered | 14 | do not promote WLD long; fresh fill audit hit the stop |
| broad-fill-audit-wld-long-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | WLD |  | long | -589.9100 | -664.2264 | stop_triggered | 59 | do not promote WLD long; fresh fill audit hit the stop |
| broad-fill-audit-near-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | NEAR |  | long | -47.8535 | -138.4991 | stop_triggered | 14 | do not promote NEAR long; fresh fill audit hit the stop |
| broad-fill-audit-near-long-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | NEAR |  | long | -167.0271 | -181.2911 | stop_triggered | 59 | do not promote NEAR long; fresh fill audit hit the stop |
| broad-fill-audit-btc-short-50bps-stop | 15m | ready | paper_fill_audit_loss | BTC |  | short | -19.6884 | -24.0880 | stop_survived | 14 | do not promote BTC short; fresh fill audit failed at 15m |
| broad-fill-audit-btc-short-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | BTC |  | short | -63.1988 | -73.7840 | stop_triggered | 59 | do not promote BTC short; fresh fill audit hit the stop |
| broad-fill-audit-eigen-short-50bps-stop | 15m | ready | paper_fill_audit_loss | EIGEN |  | short | -21.8221 | -43.5493 | stop_survived | 14 | do not promote EIGEN short; fresh fill audit failed at 15m |
| broad-fill-audit-eigen-short-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | EIGEN |  | short | -75.9631 | -118.8547 | stop_triggered | 59 | do not promote EIGEN short; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -156.4284 | -277.4606 | stop_triggered | 59 | do not promote BEAT long; fresh fill audit hit the stop |
