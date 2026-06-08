# Current Token Unlock Actionability

This separates scheduled supply events from tradable candidates. Without event-window labels, an unlock is context, not an alpha candidate.

| symbol | status | action | score | ticket | in | value USD | % supply | funding | volume | impact | reason |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HYPE | unlock_event_label_pending | create_event_window_label | 50.0999 | paper_short_candidate 32.87 | 27 | 594900000 | 4.5000 | 0.28021050 | 902576042 | 0.00011412 | unlock short thesis has no event-window forward label yet |
| ZRO | unlock_event_label_pending | create_event_window_label | 48.0764 | paper_short_candidate 31.50 | 11 | 34450000 | 10.2000 | 0.10950000 | 2627083 | 0.00045573 | unlock short thesis has no event-window forward label yet |
| KAITO | unlock_event_label_pending | create_event_window_label | 45.3183 | paper_short_candidate 27.86 | 11 | 7950000 | 7.3000 | 0.01591955 | 194367 | 0.00190502 | unlock short thesis has no event-window forward label yet |
| ME | unlock_event_crowded_squeeze_watch | label_before_short | 43.4374 | crowded_short_risk 29.04 | 1 | 10360000 | 30.9000 | -0.38971225 | 297793 | 0.00351831 | supply shock overlaps crowded short or negative funding risk |
| EIGEN | unlock_event_label_pending | create_event_window_label | 42.8040 | paper_short_candidate 25.32 | 23 | 7750000 | 5.0000 | 0.10950000 | 1154120 | 0.00215401 | unlock short thesis has no event-window forward label yet |
| PIXEL | unlock_event_not_tradeable | do_not_probe | 16.3475 | too_illiquid 12.47 | 10 | 630970 | 11.8000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| AI | unlock_event_not_tradeable | do_not_probe | 14.9462 | too_illiquid 12.97 | 22 | 444360 | 12.7000 | 0.00000000 | 0 | 0.00000000 | perp venue volume is too low for a paper probe |
| SOPH | unlock_event_not_tradeable | do_not_probe | 10.0037 | too_illiquid 5.33 | 20 | 1210000 | 5.2000 | 0.10950000 | 25689 | 0.00323102 | perp venue volume is too low for a paper probe |
