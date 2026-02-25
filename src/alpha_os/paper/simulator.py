"""Backfill simulator — run paper trading over historical dates."""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..alpha.registry import AlphaRegistry, AlphaState
from ..config import Config, DATA_DIR
from ..data.store import DataStore
from ..execution.paper import PaperExecutor
from ..forward.tracker import ForwardTracker
from ..governance.audit_log import AuditLog
from .tracker import PaperPortfolioTracker
from .trader import PaperTrader

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    n_days: int
    initial_capital: float
    final_value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    best_day: tuple[str, float]   # (date, return)
    worst_day: tuple[str, float]  # (date, return)


def run_backfill(
    asset: str,
    config: Config,
    start_date: str,
    end_date: str,
    registry_db: Path | None = None,
) -> SimulationResult:
    """Run paper trading simulation over a historical date range.

    Uses cached data only (no API sync). Requires prior data population
    via `evolve` or `paper --once`.
    """
    # Get trading dates from cache
    store = DataStore(DATA_DIR / "alpha_cache.db")
    from ..data.universe import price_signal, MACRO_SIGNALS
    try:
        price_sig = price_signal(asset)
    except KeyError:
        price_sig = asset.lower()
    features = [price_sig] + MACRO_SIGNALS

    matrix = store.get_matrix(features, start=start_date, end=end_date)
    store.close()

    if len(matrix) < 2:
        raise RuntimeError(
            f"Insufficient cached data for {start_date} to {end_date} "
            f"({len(matrix)} rows). Run `evolve` or `paper --once` first."
        )

    trading_dates = [str(d) for d in matrix.index]
    # Need at least 2 days (previous day for signal)
    trading_dates = trading_dates[1:]
    logger.info("Simulation: %d trading days (%s to %s)", len(trading_dates), trading_dates[0], trading_dates[-1])

    # Create isolated environment with temp DBs
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy registry to temp so we can reset all alphas to ACTIVE
        # without modifying the real registry
        reg_path = registry_db or DATA_DIR / "alpha_registry.db"
        sim_reg_path = tmp_path / "registry.db"
        shutil.copy2(reg_path, sim_reg_path)
        registry = AlphaRegistry(sim_reg_path)

        # Reset all alphas to ACTIVE for simulation
        all_states = [AlphaState.ACTIVE, AlphaState.PROBATION, AlphaState.RETIRED]
        n_activated = 0
        for state in all_states:
            for record in registry.list_by_state(state):
                if record.state != AlphaState.ACTIVE:
                    registry.update_state(record.alpha_id, AlphaState.ACTIVE)
                    n_activated += 1
        active = registry.list_by_state(AlphaState.ACTIVE)
        logger.info(
            "Simulation registry: %d ACTIVE alphas (%d reactivated)",
            len(active), n_activated,
        )
        if not active:
            raise RuntimeError("No alphas in registry. Run `evolve` first.")

        tracker = PaperPortfolioTracker(tmp_path / "paper.db")
        fwd_tracker = ForwardTracker(tmp_path / "fwd.db")
        executor = PaperExecutor(initial_cash=config.trading.initial_capital)
        audit_log = AuditLog(tmp_path / "audit.jsonl")
        # Reuse the real cache for data
        data_store = DataStore(DATA_DIR / "alpha_cache.db")

        trader = PaperTrader(
            asset=asset,
            config=config,
            registry=registry,
            portfolio_tracker=tracker,
            forward_tracker=fwd_tracker,
            executor=executor,
            audit_log=audit_log,
            store=data_store,
        )

        # Run simulation
        for i, dt in enumerate(trading_dates):
            result = trader.run_cycle(simulation_date=dt)
            if (i + 1) % 50 == 0 or i == len(trading_dates) - 1:
                logger.info(
                    "  Day %d/%d (%s): $%.2f (%.2f%%)",
                    i + 1, len(trading_dates), dt,
                    result.portfolio_value, result.daily_return * 100,
                )

        # Collect results
        summary = tracker.summary()
        snapshots = tracker.get_all_snapshots()

        trader.close()

    if summary is None:
        raise RuntimeError("No snapshots produced during simulation")

    # Best/worst days
    best_date, best_ret = "", 0.0
    worst_date, worst_ret = "", 0.0
    wins = 0
    for snap in snapshots:
        if snap.daily_return > best_ret:
            best_ret = snap.daily_return
            best_date = snap.date
        if snap.daily_return < worst_ret:
            worst_ret = snap.daily_return
            worst_date = snap.date
        if snap.daily_return > 0:
            wins += 1

    win_rate = wins / len(snapshots) if snapshots else 0.0

    return SimulationResult(
        n_days=summary.n_days,
        initial_capital=summary.initial_value,
        final_value=summary.final_value,
        total_return=summary.total_return,
        sharpe=summary.sharpe,
        max_drawdown=summary.max_drawdown,
        total_trades=summary.total_trades,
        win_rate=win_rate,
        best_day=(best_date, best_ret),
        worst_day=(worst_date, worst_ret),
    )
