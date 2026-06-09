# Current Token Unlock Actionability

This separates scheduled supply events from tradable candidates. Without event-window labels, an unlock is context, not an alpha candidate.

| symbol | status | action | score | ticket | in | value USD | % supply | funding | volume | impact | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | unlock_event_label_pending | create_event_window_label | 50.1975 | paper_short_candidate 32.71 | 27 | 594900000 | 4.5000 | 0.10950000 | 1042653110 | 0.00001575 | unlock short thesis has no event-window forward label yet |
| ZRO | unlock_event_label_pending | create_event_window_label | 48.1315 | paper_short_candidate 31.53 | 11 | 34450000 | 10.2000 | 0.10950000 | 2830814 | 0.00041077 | unlock short thesis has no event-window forward label yet |
| ME | unlock_event_crowded_squeeze_watch | label_before_short | 44.5542 | crowded_short_risk 28.91 | 1 | 10360000 | 30.9000 | -0.14802823 | 347465 | 0.00241702 | supply shock overlaps crowded short or negative funding risk |
| EIGEN | unlock_event_label_pending | create_event_window_label | 43.4395 | paper_short_candidate 25.40 | 22 | 7750000 | 5.0000 | 0.10950000 | 1128644 | 0.00163221 | unlock short thesis has no event-window forward label yet |
| KAITO | unlock_event_crowded_squeeze_watch | label_before_short | 39.4372 | crowded_short_risk 4.93 | 11 | 7950000 | 7.3000 | -0.07434612 | 189701 | 0.00178706 | supply shock overlaps crowded short or negative funding risk |
| PIXEL | unlock_event_not_tradeable | do_not_probe | 16.3475 | too_illiquid 12.47 | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| AI | unlock_event_not_tradeable | do_not_probe | 14.9462 | too_illiquid 12.97 | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| SOPH | unlock_event_not_tradeable | do_not_probe | 8.3681 | too_illiquid 5.17 | 20 | 1210000 | 5.2000 | 0.10950000 | 25583 | 0.00486662 | perp venue volume is too low for a paper probe |
