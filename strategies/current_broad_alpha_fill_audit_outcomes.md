# Current Broad Alpha Fill Audit Outcomes

These outcomes check fresh broad paper fill-audit tickets against public 1m candle path. They do not prove live fill quality.

| ticket | horizon | status | outcome | asset | venue | side | close | adverse | stop | candles | next step |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| broad-fill-audit-zec-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZEC |  | long | -25.8696 | -53.6634 | stop_triggered | 14 | do not promote ZEC long; fresh fill audit hit the stop |
| broad-fill-audit-zec-long-50bps-stop | 1h | pending | pending | ZEC |  | long | 118.6582 | -53.6634 | stop_triggered | 28 | wait for 1h fill-audit checkpoint for ZEC long |
| broad-fill-audit-pol-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | POL |  | short | -72.0379 | -75.5499 | stop_triggered | 14 | do not promote POL short; fresh fill audit hit the stop |
| broad-fill-audit-pol-short-50bps-stop | 1h | pending | pending | POL |  | short | -104.0551 | -104.5538 | stop_triggered | 39 | wait for 1h fill-audit checkpoint for POL short |
| broad-fill-audit-wld-short-50bps-stop | 15m | pending | pending | WLD |  | short | -27.4695 | -374.0546 | stop_triggered | 2 | wait for 15m fill-audit checkpoint for WLD short |
| broad-fill-audit-wld-short-50bps-stop | 1h | pending | pending | WLD |  | short | -27.4695 | -374.0546 | stop_triggered | 2 | wait for 1h fill-audit checkpoint for WLD short |
| broad-fill-audit-eth-long-50bps-stop | 15m | pending | pending | ETH |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for ETH |
| broad-fill-audit-eth-long-50bps-stop | 1h | pending | pending | ETH |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for ETH |
| broad-fill-audit-inj-long-50bps-stop | 15m | pending | pending | INJ |  | long | 74.7819 | 0.0000 | stop_survived | 2 | wait for 15m fill-audit checkpoint for INJ long |
| broad-fill-audit-inj-long-50bps-stop | 1h | pending | pending | INJ |  | long | 74.7819 | 0.0000 | stop_survived | 2 | wait for 1h fill-audit checkpoint for INJ long |
| broad-fill-audit-zro-short-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | ZRO |  | short | -54.5720 | -65.0470 | stop_triggered | 14 | do not promote ZRO short; fresh fill audit hit the stop |
| broad-fill-audit-zro-short-50bps-stop | 1h | pending | pending | ZRO |  | short | -110.2611 | -170.0602 | stop_triggered | 18 | wait for 1h fill-audit checkpoint for ZRO short |
| broad-fill-audit-sui-long-50bps-stop | 15m | pending | pending | SUI |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for SUI |
| broad-fill-audit-sui-long-50bps-stop | 1h | pending | pending | SUI |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for SUI |
| broad-fill-audit-fet-long-50bps-stop | 15m | pending | pending | FET |  | long | 4.6432 | 0.0000 | stop_survived | 2 | wait for 15m fill-audit checkpoint for FET long |
| broad-fill-audit-fet-long-50bps-stop | 1h | pending | pending | FET |  | long | 4.6432 | 0.0000 | stop_survived | 2 | wait for 1h fill-audit checkpoint for FET long |
| broad-fill-audit-hype-long-50bps-stop | 15m | ready | paper_fill_audit_win | HYPE |  | long | 7.3713 | -11.4490 | stop_survived | 14 | compare HYPE long fill audit against prior broad paper path before promotion |
| broad-fill-audit-hype-long-50bps-stop | 1h | pending | pending | HYPE |  | long | 41.5615 | -11.4490 | stop_survived | 28 | wait for 1h fill-audit checkpoint for HYPE long |
| broad-fill-audit-sol-long-50bps-stop | 15m | pending | pending | SOL |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for SOL |
| broad-fill-audit-sol-long-50bps-stop | 1h | pending | pending | SOL |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for SOL |
| broad-fill-audit-allo-long-50bps-stop | 15m | pending | pending | ALLO | OKX | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for ALLO |
| broad-fill-audit-allo-long-50bps-stop | 1h | pending | pending | ALLO | OKX | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for ALLO |
| broad-fill-audit-apt-long-50bps-stop | 15m | pending | pending | APT |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for APT |
| broad-fill-audit-apt-long-50bps-stop | 1h | pending | pending | APT |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for APT |
| broad-fill-audit-btc-long-50bps-stop | 15m | pending | pending | BTC |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 15m fill-audit checkpoint for BTC |
| broad-fill-audit-btc-long-50bps-stop | 1h | pending | pending | BTC |  | long | 0.0000 | 0.0000 | not_checked | 0 | wait for 1h fill-audit checkpoint for BTC |
| broad-fill-audit-chip-long-50bps-stop | 15m | ready | paper_fill_audit_win | CHIP |  | long | 97.6187 | -11.8326 | stop_survived | 14 | compare CHIP long fill audit against prior broad paper path before promotion |
| broad-fill-audit-chip-long-50bps-stop | 1h | pending | pending | CHIP |  | long | 160.3313 | -11.8326 | stop_survived | 28 | wait for 1h fill-audit checkpoint for CHIP long |
| broad-fill-audit-wld-long-50bps-stop | 15m | pending | pending | WLD |  | long | 27.5452 | -6.8401 | stop_survived | 2 | wait for 15m fill-audit checkpoint for WLD long |
| broad-fill-audit-wld-long-50bps-stop | 1h | pending | pending | WLD |  | long | 27.5452 | -6.8401 | stop_survived | 2 | wait for 1h fill-audit checkpoint for WLD long |
| broad-fill-audit-near-long-50bps-stop | 15m | pending | pending | NEAR |  | long | -15.1843 | -15.1843 | stop_survived | 2 | wait for 15m fill-audit checkpoint for NEAR long |
| broad-fill-audit-near-long-50bps-stop | 1h | pending | pending | NEAR |  | long | -15.1843 | -15.1843 | stop_survived | 2 | wait for 1h fill-audit checkpoint for NEAR long |
| broad-fill-audit-btc-short-50bps-stop | 15m | ready | paper_fill_audit_loss | BTC |  | short | -19.6884 | -24.0880 | stop_survived | 14 | do not promote BTC short; fresh fill audit failed at 15m |
| broad-fill-audit-btc-short-50bps-stop | 1h | pending | pending | BTC |  | short | -50.2481 | -66.4702 | stop_triggered | 28 | wait for 1h fill-audit checkpoint for BTC short |
| broad-fill-audit-eigen-short-50bps-stop | 15m | ready | paper_fill_audit_loss | EIGEN |  | short | -21.8221 | -43.5493 | stop_survived | 14 | do not promote EIGEN short; fresh fill audit failed at 15m |
| broad-fill-audit-eigen-short-50bps-stop | 1h | pending | pending | EIGEN |  | short | -92.0910 | -118.8547 | stop_triggered | 28 | wait for 1h fill-audit checkpoint for EIGEN short |
| broad-fill-audit-beat-long-50bps-stop | 15m | ready | paper_fill_audit_stop_loss | BEAT | OKX | long | -123.3158 | -220.3699 | stop_triggered | 14 | do not promote BEAT long; fresh fill audit hit the stop |
| broad-fill-audit-beat-long-50bps-stop | 1h | pending | pending | BEAT | OKX | long | -51.6100 | -277.4606 | stop_triggered | 45 | wait for 1h fill-audit checkpoint for BEAT long |
