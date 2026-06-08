# Current Token Unlock Actionability

This separates scheduled supply events from tradable candidates. Without event-window labels, an unlock is context, not an alpha candidate.

| symbol | status | action | score | ticket | in | value USD | % supply | funding | volume | impact | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | unlock_event_label_pending | create_event_window_label | 50.0480 | paper_short_candidate 32.69 | 27 | 594900000 | 4.5000 | 0.10950000 | 760191800 | 0.00017575 | unlock short thesis has no event-window forward label yet |
| ZRO | unlock_event_label_pending | create_event_window_label | 47.8988 | paper_short_candidate 31.51 | 11 | 34450000 | 10.2000 | 0.10950000 | 2894196 | 0.00064664 | unlock short thesis has no event-window forward label yet |
| KAITO | unlock_event_label_pending | create_event_window_label | 45.4357 | paper_short_candidate 27.96 | 11 | 7950000 | 7.3000 | 0.10950000 | 183414 | 0.00178823 | unlock short thesis has no event-window forward label yet |
| ME | unlock_event_crowded_squeeze_watch | label_before_short | 45.1408 | crowded_short_risk 29.42 | 1 | 10360000 | 30.9000 | -0.61421528 | 187325 | 0.00183885 | supply shock overlaps crowded short or negative funding risk |
| EIGEN | unlock_event_label_pending | create_event_window_label | 42.7704 | paper_short_candidate 25.30 | 23 | 7750000 | 5.0000 | 0.10950000 | 1017780 | 0.00217984 | unlock short thesis has no event-window forward label yet |
| PIXEL | unlock_event_not_tradeable | do_not_probe | 16.3475 | too_illiquid 12.47 | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| AI | unlock_event_not_tradeable | do_not_probe | 14.9462 | too_illiquid 12.97 | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| SOPH | unlock_event_not_tradeable | do_not_probe | 8.0942 | too_illiquid 5.14 | 20 | 1210000 | 5.2000 | 0.10950000 | 23193 | 0.00514044 | perp venue volume is too low for a paper probe |
