# Large Module Boundary

## Problem

Several modules have grown into broad containers for unrelated responsibilities.
Large files are not automatically wrong, but these files are now hiding domain
boundaries and making it harder to see which code is still needed.

Current size markers:

- none currently blocking

Removed:

- `src/alpha_os/decision_backtest.py`: removed because it was a signal-first
  portfolio-decision rollout adapter, not the desired TradingStrategy rollout
  model.
- `src/alpha_os/strategy_backtest_evaluation.py`: removed after the old direct
  strategy evaluation path became unused.
- `src/alpha_os/portfolio_sizing_policy.py`: removed because it was a legacy
  rich allocator path, not the black-box `TradingStrategy` contract.

Large test files also indicate mixed workflows:

- `tests/test_alpha_os_signal_discovery_run_evaluation.py`: 160 lines

## Risk

These modules make dead paths look alive and encourage new behavior to be added
to the nearest large file instead of the right boundary.

The main risk is preserving old abstractions by splitting files mechanically.
Do not treat this as a formatting or folder-organization task. First remove
unused behavior and stale concepts, then split only when the remaining
responsibilities are clear.

## Boundary

Prefer deletion before extraction.

When a large module is touched, identify whether its responsibilities are:

- active domain behavior
- adapter or persistence code
- evaluation/reporting code
- diagnostics or debugging support
- legacy compatibility

Only extract code after those categories are clear. If a responsibility is no
longer needed, remove it instead of moving it.

## Suggested Order

Continue with these candidates:

- none currently selected

## Close Condition

Close this when the largest modules have either been reduced to coherent
responsibilities or have follow-up issues for the specific boundaries that need
separate work.
