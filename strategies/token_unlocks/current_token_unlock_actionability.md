# Current Token Unlock Actionability

This separates scheduled supply events from tradable candidates. Without event-window labels, an unlock is context, not an alpha candidate.

| symbol | status | action | score | ticket | in | value USD | % supply | funding | volume | impact | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | unlock_event_label_pending | create_event_window_label | 50.0984 | paper_short_candidate 32.70 | 27 | 594900000 | 4.5000 | 0.10950000 | 881357291 | 0.00012984 | unlock short thesis has no event-window forward label yet |
| ZRO | unlock_event_label_pending | create_event_window_label | 48.2472 | paper_short_candidate 31.52 | 11 | 34450000 | 10.2000 | 0.10950000 | 2604693 | 0.00028373 | unlock short thesis has no event-window forward label yet |
| KAITO | unlock_event_label_pending | create_event_window_label | 45.1427 | paper_short_candidate 27.93 | 11 | 7950000 | 7.3000 | 0.10950000 | 195800 | 0.00207866 | unlock short thesis has no event-window forward label yet |
| ME | unlock_event_crowded_squeeze_watch | label_before_short | 44.5630 | crowded_short_risk 29.71 | 1 | 10360000 | 30.9000 | -0.95318962 | 298745 | 0.00240770 | supply shock overlaps crowded short or negative funding risk |
| EIGEN | unlock_event_label_pending | create_event_window_label | 43.3330 | paper_short_candidate 25.38 | 23 | 7750000 | 5.0000 | 0.10950000 | 1210527 | 0.00162690 | unlock short thesis has no event-window forward label yet |
| PIXEL | unlock_event_not_tradeable | do_not_probe | 16.3475 | too_illiquid 12.47 | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| AI | unlock_event_not_tradeable | do_not_probe | 14.9462 | too_illiquid 12.97 | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| SOPH | unlock_event_not_tradeable | do_not_probe | 9.6215 | too_illiquid 5.30 | 20 | 1210000 | 5.2000 | 0.10950000 | 27826 | 0.00361337 | perp venue volume is too low for a paper probe |
