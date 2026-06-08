# Current Token Unlock Actionability

This separates scheduled supply events from tradable candidates. Without event-window labels, an unlock is context, not an alpha candidate.

| symbol | status | action | score | ticket | in | value USD | % supply | funding | volume | impact | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | unlock_event_label_pending | create_event_window_label | 49.7649 | paper_short_candidate 32.64 | 28 | 594900000 | 4.5000 | 0.10950000 | 766770426 | 0.00032840 | unlock short thesis has no event-window forward label yet |
| ZRO | unlock_event_label_pending | create_event_window_label | 47.5116 | paper_short_candidate 31.45 | 12 | 34450000 | 10.2000 | 0.10950000 | 2902449 | 0.00089583 | unlock short thesis has no event-window forward label yet |
| KAITO | unlock_event_label_pending | create_event_window_label | 45.4224 | paper_short_candidate 27.94 | 12 | 7950000 | 7.3000 | 0.10950000 | 184036 | 0.00166419 | unlock short thesis has no event-window forward label yet |
| ME | unlock_event_crowded_squeeze_watch | label_before_short | 44.2810 | crowded_short_risk 29.42 | 2 | 10360000 | 30.9000 | -0.70236716 | 187445 | 0.00235373 | supply shock overlaps crowded short or negative funding risk |
| EIGEN | unlock_event_label_pending | create_event_window_label | 42.2281 | paper_short_candidate 25.25 | 23 | 7750000 | 5.0000 | 0.10950000 | 1010223 | 0.00272183 | unlock short thesis has no event-window forward label yet |
| PIXEL | unlock_event_not_tradeable | do_not_probe | 16.2091 | too_illiquid 12.44 | 11 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| AI | unlock_event_not_tradeable | do_not_probe | 14.8116 | too_illiquid 12.94 | 23 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| SOPH | unlock_event_not_tradeable | do_not_probe | 5.5112 | too_illiquid 4.88 | 20 | 1210000 | 5.2000 | 0.10950000 | 23223 | 0.00772343 | perp venue volume is too low for a paper probe |
