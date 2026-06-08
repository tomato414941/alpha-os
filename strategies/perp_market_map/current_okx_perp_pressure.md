# Current OKX Perp Pressure

This maps current OKX USDT swap funding, premium, open interest, volume, and near-touch spread. It is a candidate screen, not a deployable strategy.

| asset | action | ann funding | settled ann funding | premium | OI USD | volume USD | OI/vol | spread bps | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LAYER | long_carry_discount_watch | -8.891676 | -0.381355 | -0.008283 | 1396072 | 22078131 | 0.0632 | 2.5589 | 1203.764269 |
| HOME | long_carry_discount_watch | -5.474308 | -4.196914 | -0.041420 | 2610130 | 42867577 | 0.0609 | 3.5430 | 804.241061 |
| BEAT | short_carry_premium_watch | 0.816555 | 1.112828 | 0.006308 | 42411650 | 769654995 | 0.0551 | 1.8158 | 166.020450 |
| H | long_carry_discount_watch | -1.324073 | -0.282734 | -0.001705 | 9280304 | 64510602 | 0.1439 | 11.8865 | 133.363725 |
| BZ | long_carry_discount_watch | -1.038068 | -0.613949 | -0.002746 | 5477914 | 30936674 | 0.1771 | 1.0673 | 124.324848 |
| CL | long_carry_discount_watch | -0.789510 | -0.484313 | -0.002365 | 25819653 | 173275156 | 0.1490 | 1.1013 | 105.208083 |
| ZEC | long_carry_discount_watch | -0.652026 | -0.917791 | -0.001204 | 58738314 | 778075420 | 0.0755 | 0.2187 | 72.145948 |
| LAB | short_carry_premium_watch | 0.350045 | 0.054750 | 0.001886 | 17050110 | 108506639 | 0.1571 | 0.8130 | 39.513276 |
| RAVE | short_carry_premium_watch | 0.350006 | 0.284372 | 0.001023 | 3641032 | 12481734 | 0.2917 | 2.7522 | 24.604044 |
| NVDA | short_carry_premium_watch | 0.307606 | 0.266403 | 0.000932 | 11455086 | 32146740 | 0.3563 | 0.4804 | 23.893682 |
| PIPPIN | short_carry_premium_watch | 0.345620 | 0.077081 | 0.000288 | 6549572 | 212950685 | 0.0308 | 3.7140 | 22.405681 |
| TRX | long_carry_discount_watch | -0.285288 | -0.220443 | -0.000879 | 14548940 | 11666972 | 1.2470 | 0.3067 | 20.784062 |
| ICP | long_carry_discount_watch | -0.338728 | -0.293832 | -0.000491 | 4949486 | 13176759 | 0.3756 | 4.2328 | 20.067773 |
| AAVE | long_carry_discount_watch | -0.308793 | -0.274875 | -0.000533 | 9357456 | 21881119 | 0.4276 | 1.5602 | 19.996107 |
| ETH | long_carry_discount_watch | -0.148091 | -0.080872 | -0.000758 | 1186802224 | 8925876933 | 0.1330 | 0.0593 | 18.440690 |
| TAO | long_carry_discount_watch | -0.270663 | -0.117320 | -0.000591 | 11074455 | 25207363 | 0.4393 | 4.6009 | 18.235191 |
| EWY | short_carry_premium_watch | 0.312151 | 0.373704 | 0.000526 | 2107331 | 19782240 | 0.1065 | 0.5386 | 18.185531 |
| BSB | short_carry_watch | 0.212591 | 0.321139 | -0.000969 | 6339466 | 188897375 | 0.0336 | 3.1061 | 17.736378 |
| DOT | long_carry_discount_watch | -0.185803 | 0.051070 | -0.001009 | 7459383 | 19258965 | 0.3873 | 10.1061 | 13.894582 |
| ORCL | short_carry_premium_watch | 0.203574 | 0.004855 | 0.000312 | 1922154 | 11244641 | 0.1709 | 5.1851 | 10.374035 |
| SAHARA | short_carry_premium_watch | 0.157910 | 0.310260 | 0.000409 | 6021191 | 26113860 | 0.2306 | 2.8349 | 9.536986 |
| HYPE | short_carry_watch | 0.104667 | 0.109500 | -0.000457 | 96303727 | 606084476 | 0.1589 | 1.5762 | 9.000506 |
| CRCL | short_carry_premium_watch | 0.145280 | 0.170508 | 0.000386 | 9497713 | 26828824 | 0.3540 | 1.2177 | 8.972400 |
| RENDER | long_carry_discount_watch | -0.165323 | -0.150994 | -0.000385 | 2956136 | 10026238 | 0.2948 | 6.0006 | 8.872604 |
| LINK | long_carry_discount_watch | -0.120769 | -0.023745 | -0.000620 | 20959961 | 42249543 | 0.4961 | 1.2405 | 8.819376 |

## Interpretation

High scores mean the instrument is liquid enough to inspect and has large current funding or premium pressure. This does not include future-return labels, funding decay labels, fees, maker/taker fill probability, or liquidation data.
