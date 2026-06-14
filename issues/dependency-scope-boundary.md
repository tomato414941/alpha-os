# Dependency Scope Boundary

## Problem

The current package dependency list still reflects removed backtest, optimizer,
exchange, research, and Polymarket code.

The maintained source code is now centered on the `TradingStrategy` contract,
but `pyproject.toml` and `uv.lock` still carry heavy dependencies such as:

- `numpy`
- `pandas`
- `requests`
- `scipy`
- `cvxpy`
- `skfolio`
- `ccxt`
- `deap`
- `ribs`
- `py-clob-client`
- `web3`

That makes the project look larger and more coupled than the current code is.

## Risk

Stale dependencies can hide old design assumptions.

They also make installation, CI, and future design discussions depend on
libraries that no maintained runtime path currently needs.

## Direction

Do not clean this up immediately.

When this issue is picked up, compare each dependency against maintained source
code and checked-in experiments. Remove runtime and optional dependencies that
only supported deleted alpha-os implementation paths.

Keep `dev` dependencies only when they are still used by the active toolchain.

## Close Condition

Close this when `pyproject.toml` and `uv.lock` match the maintained codebase
instead of historical implementation paths.
