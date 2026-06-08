# Current Surviving Alpha Exit Regime Candidates

This searches stop/exit regimes for second-repeat survivors that had path-risk issues. It is a paper path review, not a live execution rule.

| candidate | asset | decision | horizon | status | close | adverse | stop50 | stop100 | priority | next step |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| chip-long-30m-exit | CHIP | paper_long | 30 | wide_stop_exit_candidate | 256.4181 | -57.4254 | stop_would_trigger | stop_survived | 292.5009 | paper-repeat CHIP paper_long with 30m exit and 100bps stop on a fresh trigger |
| chip-long-240m-exit | CHIP | paper_long | 240 | wide_stop_exit_candidate | 156.3076 | -74.6223 | stop_would_trigger | stop_survived | 193.7723 | paper-repeat CHIP paper_long with 240m exit and 100bps stop on a fresh trigger |
| chip-long-60m-exit | CHIP | paper_long | 60 | wide_stop_exit_candidate | 102.5673 | -57.4254 | stop_would_trigger | stop_survived | 148.6304 | paper-repeat CHIP paper_long with 60m exit and 100bps stop on a fresh trigger |
| chip-long-120m-exit | CHIP | paper_long | 120 | wide_stop_low_edge_watch | 0.9213 | -57.4254 | stop_would_trigger | stop_survived | 46.9844 | keep CHIP paper_long as context; edge is positive but too small for a new repeat |
| chip-long-10m-exit | CHIP | paper_long | 10 | wide_stop_low_edge_watch | 19.6536 | -57.4254 | stop_would_trigger | stop_survived | -4.1457 | keep CHIP paper_long as context; edge is positive but too small for a new repeat |
| chip-long-15m-exit | CHIP | paper_long | 15 | exit_horizon_negative | -19.6536 | -57.4254 | stop_would_trigger | stop_survived | -23.7993 | do not use 15m exit for CHIP paper_long; close return was negative |
| chip-long-5m-exit | CHIP | paper_long | 5 | exit_horizon_negative | -24.8741 | -57.4254 | stop_would_trigger | stop_survived | -28.7127 | do not use 5m exit for CHIP paper_long; close return was negative |
