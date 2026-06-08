# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-zec-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZEC |  | long | -25.8696 | -53.6634 | stop_triggered | 14 | do not promote ZEC long; fresh fill audit hit the stop |
| broad-fill-audit-zec-long-50bps-stop | 1h | pending | pending | ZEC |  | long | 8.1243 | -86.8022 | stop_triggered | 45 | wait for 1h fill-audit checkpoint for ZEC long |
| broad-fill-audit-pol-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | POL |  | short | -72.0379 | -75.5499 | stop_triggered | 14 | do not promote POL short; fresh fill audit hit the stop |
| broad-fill-audit-pol-short-50bps-stop | 1h | pending | pending | POL |  | short | -96.1951 | -104.5538 | stop_triggered | 57 | wait for 1h fill-audit checkpoint for POL short |
| broad-fill-audit-wld-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | WLD |  | short | 437.0225 | -374.0546 | stop_triggered | 14 | do not promote WLD short; fresh fill audit hit the stop |
| broad-fill-audit-wld-short-50bps-stop | 1h | pending | pending | WLD |  | short | 536.8448 | -374.0546 | stop_triggered | 19 | wait for 1h fill-audit checkpoint for WLD short |
| broad-fill-audit-eth-long-50bps-stop | 15m | ready | paper_fill_audit_loss | ETH |  | long | -23.3686 | -46.7372 | stop_survived | 14 | do not promote ETH long; fresh fill audit failed at 15m |
| broad-fill-audit-eth-long-50bps-stop | 1h | pending | pending | ETH |  | long | -19.8633 | -46.7372 | stop_survived | 16 | wait for 1h fill-audit checkpoint for ETH long |
| broad-fill-audit-inj-long-50bps-stop | 15m | ready | paper_fill_audit_win | INJ |  | long | 98.4974 | -35.6599 | stop_survived | 14 | compare INJ long fill audit against prior broad paper path before promotion |
| broad-fill-audit-inj-long-50bps-stop | 1h | pending | pending | INJ |  | long | 105.0755 | -35.6599 | stop_survived | 19 | wait for 1h fill-audit checkpoint for INJ long |
| broad-fill-audit-zro-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZRO |  | short | -54.5720 | -65.0470 | stop_triggered | 14 | do not promote ZRO short; fresh fill audit hit the stop |
| broad-fill-audit-zro-short-50bps-stop | 1h | pending | pending | ZRO |  | short | -219.4547 | -219.4547 | stop_triggered | 35 | wait for 1h fill-audit checkpoint for ZRO short |
| broad-fill-audit-sui-long-50bps-stop | 15m | ready | paper_fill_audit_loss | SUI |  | long | -10.4194 | -46.4965 | stop_survived | 14 | do not promote SUI long; fresh fill audit failed at 15m |
| broad-fill-audit-sui-long-50bps-stop | 1h | pending | pending | SUI |  | long | -25.2670 | -46.4965 | stop_survived | 16 | wait for 1h fill-audit checkpoint for SUI long |
| broad-fill-audit-fet-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | FET |  | long | -18.5727 | -86.3630 | stop_triggered | 14 | do not promote FET long; fresh fill audit hit the stop |
| broad-fill-audit-fet-long-50bps-stop | 1h | pending | pending | FET |  | long | -41.3242 | -86.3630 | stop_triggered | 19 | wait for 1h fill-audit checkpoint for FET long |
| broad-fill-audit-hype-long-50bps-stop | 15m | ready | paper_fill_audit_win | HYPE |  | long | 7.3713 | -11.4490 | stop_survived | 14 | compare HYPE long fill audit against prior broad paper path before promotion |
| broad-fill-audit-hype-long-50bps-stop | 1h | pending | pending | HYPE |  | long | 35.7585 | -11.4490 | stop_survived | 45 | wait for 1h fill-audit checkpoint for HYPE long |
| broad-fill-audit-sol-long-50bps-stop | 15m | ready | paper_fill_audit_loss | SOL |  | long | -26.4858 | -41.2001 | stop_survived | 14 | do not promote SOL long; fresh fill audit failed at 15m |
| broad-fill-audit-sol-long-50bps-stop | 1h | pending | pending | SOL |  | long | -34.2844 | -41.2001 | stop_survived | 16 | wait for 1h fill-audit checkpoint for SOL long |
| broad-fill-audit-allo-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ALLO | OKX | long | 186.9018 | -75.1616 | stop_triggered | 14 | do not promote ALLO long; fresh fill audit hit the stop |
| broad-fill-audit-allo-long-50bps-stop | 1h | pending | pending | ALLO | OKX | long | 232.7504 | -75.1616 | stop_triggered | 16 | wait for 1h fill-audit checkpoint for ALLO long |
| broad-fill-audit-apt-long-50bps-stop | 15m | ready | paper_fill_audit_loss | APT |  | long | -8.7642 | -45.2819 | stop_survived | 14 | do not promote APT long; fresh fill audit failed at 15m |
| broad-fill-audit-apt-long-50bps-stop | 1h | pending | pending | APT |  | long | -23.3713 | -45.2819 | stop_survived | 16 | wait for 1h fill-audit checkpoint for APT long |
| broad-fill-audit-btc-long-50bps-stop | 15m | ready | paper_fill_audit_loss | BTC |  | long | -10.2031 | -19.3074 | stop_survived | 14 | do not promote BTC long; fresh fill audit failed at 15m |
| broad-fill-audit-btc-long-50bps-stop | 1h | pending | pending | BTC |  | long | -9.2613 | -19.3074 | stop_survived | 16 | wait for 1h fill-audit checkpoint for BTC long |
| broad-fill-audit-chip-long-50bps-stop | 15m | ready | paper_fill_audit_win | CHIP |  | long | 97.6187 | -11.8326 | stop_survived | 14 | compare CHIP long fill audit against prior broad paper path before promotion |
| broad-fill-audit-chip-long-50bps-stop | 1h | pending | pending | CHIP |  | long | 239.0179 | -11.8326 | stop_survived | 45 | wait for 1h fill-audit checkpoint for CHIP long |
| broad-fill-audit-wld-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | WLD |  | long | -418.7233 | -664.2264 | stop_triggered | 14 | do not promote WLD long; fresh fill audit hit the stop |
| broad-fill-audit-wld-long-50bps-stop | 1h | pending | pending | WLD |  | long | -509.4929 | -664.2264 | stop_triggered | 19 | wait for 1h fill-audit checkpoint for WLD long |
| broad-fill-audit-near-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | NEAR |  | long | -47.8535 | -138.4991 | stop_triggered | 14 | do not promote NEAR long; fresh fill audit hit the stop |
| broad-fill-audit-near-long-50bps-stop | 1h | pending | pending | NEAR |  | long | -36.8104 | -138.4991 | stop_triggered | 19 | wait for 1h fill-audit checkpoint for NEAR long |
| broad-fill-audit-btc-short-50bps-stop | 15m | ready | paper_fill_audit_loss | BTC |  | short | -19.6884 | -24.0880 | stop_survived | 14 | do not promote BTC short; fresh fill audit failed at 15m |
| broad-fill-audit-btc-short-50bps-stop | 1h | pending | pending | BTC |  | short | -44.4640 | -66.4702 | stop_triggered | 45 | wait for 1h fill-audit checkpoint for BTC short |
| broad-fill-audit-eigen-short-50bps-stop | 15m | ready | paper_fill_audit_loss | EIGEN |  | short | -21.8221 | -43.5493 | stop_survived | 14 | do not promote EIGEN short; fresh fill audit failed at 15m |
| broad-fill-audit-eigen-short-50bps-stop | 1h | pending | pending | EIGEN |  | short | -81.3449 | -118.8547 | stop_triggered | 45 | wait for 1h fill-audit checkpoint for EIGEN short |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -156.4284 | -277.4606 | stop_triggered | 59 | do not promote BEAT long; fresh fill audit hit the stop |
