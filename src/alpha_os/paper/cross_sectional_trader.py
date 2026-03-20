"""Cross-sectional trader — Numerai-style cross-asset allocation."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np

from alpha_os.alpha.combiner import (
    compute_tc_scores,
    compute_tc_weights,
    cross_asset_neutralize,
    weighted_combine_scalar,
)
from alpha_os.alpha.evaluator import evaluate_expression, normalize_signal
from alpha_os.alpha.managed_alphas import ManagedAlphaStore
from alpha_os.config import Config, DATA_DIR, asset_data_dir
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.store import DataStore
from alpha_os.data.universe import build_feature_list, price_signal
from alpha_os.dsl import parse
from alpha_os.execution.executor import Executor, Fill
from alpha_os.execution.paper import PaperExecutor
from alpha_os.execution.planning import build_target_position, build_execution_intent
from alpha_os.governance.audit_log import AuditLog
from alpha_os.paper.tracker import PaperPortfolioTracker, PortfolioSnapshot
from alpha_os.risk.circuit_breaker import CircuitBreaker
from alpha_os.risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class CrossSectionalCycleResult:
    date: str
    per_asset_signals: dict[str, float]
    neutralized_signals: dict[str, float]
    fills: list[Fill]
    portfolio_value: float
    daily_pnl: float = 0.0
    daily_return: float = 0.0
    dd_scale: float = 1.0
    n_deployed_alphas: int = 0
    n_assets_traded: int = 0
    n_alphas_evaluated: int = 0


class CrossSectionalTrader:
    """Cross-asset portfolio trader.

    Evaluates deployed alphas across all tradeable assets, computes
    per-asset signals via TC weighting, neutralizes cross-sectionally,
    and allocates capital proportional to relative signal strength.
    """

    def __init__(
        self,
        tradeable_assets: list[str],
        config: Config,
        registry: ManagedAlphaStore | None = None,
        executor: Executor | None = None,
        risk_manager: RiskManager | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        store: DataStore | None = None,
    ):
        self.tradeable_assets = tradeable_assets
        self.config = config
        self.xs_config = config.cross_sectional

        # Price signal mapping: asset → signal name
        self.price_signals = {}
        for asset in tradeable_assets:
            try:
                self.price_signals[asset] = price_signal(asset)
            except KeyError:
                self.price_signals[asset] = asset.lower()

        # Shared registry (from unified generator)
        registry_asset = self.xs_config.registry_asset
        adir = asset_data_dir(registry_asset)
        self.registry = registry or ManagedAlphaStore(
            db_path=adir / "alpha_registry.db"
        )

        # Cross-sectional state directory
        xs_dir = asset_data_dir("XS")
        self.portfolio_tracker = PaperPortfolioTracker(
            db_path=xs_dir / "paper_trading.db"
        )
        self.audit_log = AuditLog(log_path=xs_dir / "audit.jsonl")
        self.circuit_breaker = circuit_breaker or CircuitBreaker.load(
            path=xs_dir / "metrics" / "circuit_breaker.json"
        )

        self.risk_manager = risk_manager or RiskManager(
            config.risk.to_manager_config()
        )

        self.initial_capital = config.trading.initial_capital
        self.max_position_pct = config.paper.max_position_pct
        self.min_trade_usd = config.paper.min_trade_usd

        self.executor = executor or PaperExecutor(
            initial_cash=self.initial_capital,
            supports_short=config.trading.supports_short,
            cost_model=config.execution.to_cost_model(),
        )

        if store is not None:
            self.store = store
        else:
            client = build_signal_client_from_config(config.api)
            self.store = DataStore(DATA_DIR / "alpha_cache.db", client)

        self._restore_state()

    def _restore_state(self) -> None:
        snapshot = self.portfolio_tracker.get_last_snapshot()
        if snapshot is None:
            self.risk_manager.reset(self.initial_capital)
            return

        # Restore managed positions (filter to known price signals)
        valid_signals = set(self.price_signals.values())
        own_positions = {
            k: v for k, v in snapshot.positions.items()
            if k in valid_signals
        }
        if isinstance(self.executor, PaperExecutor):
            self.executor._cash = snapshot.cash
            self.executor._positions = dict(own_positions)
        elif hasattr(self.executor, "_managed_cash"):
            self.executor._managed_cash = snapshot.cash
            self.executor._managed_positions = dict(own_positions)

        equity_curve = self.portfolio_tracker.get_equity_curve()
        if equity_curve:
            self.risk_manager.reset(equity_curve[0][1])
            for _date, equity in equity_curve[1:]:
                self.risk_manager.update_equity(equity)
        else:
            self.risk_manager.reset(self.initial_capital)

    def run_cycle(self) -> CrossSectionalCycleResult:
        t0 = time.perf_counter()
        self.circuit_breaker.reload()
        now = datetime.now(timezone.utc)
        cycle_key = now.strftime("%Y-%m-%dT%H:%M:%S")
        today_date = date.today().isoformat()

        if self.circuit_breaker.halted:
            logger.warning("Circuit breaker halted, skipping cycle")
            return CrossSectionalCycleResult(
                date=cycle_key, per_asset_signals={}, neutralized_signals={},
                fills=[], portfolio_value=self.executor.portfolio_value,
            )

        # 1. Load data (shared across all assets)
        features = build_feature_list(self.tradeable_assets[0])
        matrix = self.store.get_matrix(features, end=today_date)
        if matrix is None or len(matrix) < 2:
            logger.warning("Insufficient data")
            return CrossSectionalCycleResult(
                date=cycle_key, per_asset_signals={}, neutralized_signals={},
                fills=[], portfolio_value=self.executor.portfolio_value,
            )
        data = {col: matrix[col].values for col in matrix.columns}

        # 2. Get deployed alphas
        deployed_ids = self.registry.deployed_alpha_ids()
        deployed_records = [self.registry.get(aid) for aid in deployed_ids]
        deployed_records = [r for r in deployed_records if r is not None]

        if not deployed_records:
            logger.warning("No deployed alphas")
            return CrossSectionalCycleResult(
                date=cycle_key, per_asset_signals={}, neutralized_signals={},
                fills=[], portfolio_value=self.executor.portfolio_value,
            )

        # 3. Evaluate each alpha once (DSL expressions are asset-agnostic)
        n_days = len(matrix)
        alpha_signals: dict[str, float] = {}  # alpha_id → signal_yesterday
        alpha_signal_arrays: dict[str, np.ndarray] = {}
        n_evaluated = 0

        for record in deployed_records:
            try:
                expr = parse(record.expression)
                sig = evaluate_expression(expr, data, n_days)
                sig_norm = normalize_signal(sig)
                signal_yesterday = float(sig_norm[-2])
                if not np.isfinite(signal_yesterday):
                    for offset in range(3, min(10, len(sig_norm))):
                        fb = float(sig_norm[-offset])
                        if np.isfinite(fb):
                            signal_yesterday = fb
                            break
                if np.isfinite(signal_yesterday):
                    alpha_signals[record.alpha_id] = signal_yesterday
                    alpha_signal_arrays[record.alpha_id] = sig_norm
                    n_evaluated += 1
            except Exception:
                continue

        if not alpha_signals:
            logger.warning("No valid alpha signals")
            return CrossSectionalCycleResult(
                date=cycle_key, per_asset_signals={}, neutralized_signals={},
                fills=[], portfolio_value=self.executor.portfolio_value,
                n_deployed_alphas=len(deployed_records),
            )

        # 4. Per-asset TC-weighted signal
        per_asset_signals: dict[str, float] = {}
        for asset in self.tradeable_assets:
            ps = self.price_signals[asset]
            prices_arr = data.get(ps)
            if prices_arr is None or len(prices_arr) < 20:
                per_asset_signals[asset] = 0.0
                continue

            asset_returns = np.diff(prices_arr) / prices_arr[:-1]
            finite_mask = np.isfinite(asset_returns)
            clean_returns = asset_returns[finite_mask]
            if len(clean_returns) < 20:
                per_asset_signals[asset] = 0.0
                continue

            # TC for this asset's returns
            tc = compute_tc_scores(
                {aid: arr[-len(asset_returns):][finite_mask]
                 for aid, arr in alpha_signal_arrays.items()},
                clean_returns,
            )
            weights = compute_tc_weights(tc)
            combined = weighted_combine_scalar(alpha_signals, weights)
            per_asset_signals[asset] = combined

        logger.info(
            "Per-asset signals: %s (%d alphas evaluated)",
            {a: f"{v:.4f}" for a, v in per_asset_signals.items()},
            n_evaluated,
        )

        # 5. Cross-sectional neutralization
        if self.xs_config.neutralize and len(per_asset_signals) > 1:
            neutralized = cross_asset_neutralize(per_asset_signals)
        else:
            neutralized = dict(per_asset_signals)

        logger.info(
            "Neutralized signals: %s",
            {a: f"{v:.4f}" for a, v in neutralized.items()},
        )

        # 6. Risk adjustment
        prev_value = self.executor.portfolio_value
        self.risk_manager.update_equity(prev_value)
        dd_s = self.risk_manager.dd_scale

        # 7. Capital allocation
        max_per_asset = self.xs_config.max_per_asset_pct
        total_abs = sum(abs(v) for v in neutralized.values())
        fills: list[Fill] = []

        for asset in self.tradeable_assets:
            signal = neutralized.get(asset, 0.0)
            if not np.isfinite(signal) or abs(signal) < 1e-6:
                continue

            if self.xs_config.allocation_mode == "equal_weight":
                weight = 1.0 / len(self.tradeable_assets)
            else:
                weight = abs(signal) / total_abs if total_abs > 0 else 1.0 / len(self.tradeable_assets)
            weight = min(weight, max_per_asset)

            adjusted_signal = np.sign(signal) * weight * dd_s
            ps = self.price_signals[asset]
            current_price = self.executor.fetch_ticker_price(ps)
            if current_price is None or current_price <= 0:
                continue

            target = build_target_position(
                symbol=ps,
                final_signal=float(adjusted_signal),
                portfolio_value=prev_value,
                current_price=current_price,
                max_position_pct=self.max_position_pct,
                min_trade_usd=self.min_trade_usd,
                supports_short=self.executor.supports_short,
            )

            intent = build_execution_intent(
                target=target,
                current_qty=self.executor.get_position(ps),
            )
            if intent is not None:
                fill = self.executor.execute_intent(intent)
                if fill is not None:
                    fills.append(fill)

        # 8. Snapshot
        portfolio_value = self.executor.portfolio_value
        daily_pnl = portfolio_value - prev_value
        daily_return = daily_pnl / prev_value if prev_value > 0 else 0.0

        combined_signal = sum(neutralized.values()) / len(neutralized) if neutralized else 0.0
        snapshot = PortfolioSnapshot(
            date=cycle_key,
            cash=self.executor.get_cash(),
            positions=self.executor.all_positions,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            combined_signal=combined_signal,
            dd_scale=dd_s,
            vol_scale=1.0,
        )
        self.portfolio_tracker.save_snapshot(snapshot)
        self.portfolio_tracker.save_fills(cycle_key, fills)

        self.circuit_breaker.record_trade(daily_pnl)

        for fill in fills:
            self.audit_log.log_trade(
                "xs_portfolio", fill.symbol, fill.side, fill.qty, fill.price,
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "XS cycle %s: %.1fs, $%.2f, %d fills, %d assets",
            cycle_key, elapsed, portfolio_value, len(fills), len(self.tradeable_assets),
        )

        return CrossSectionalCycleResult(
            date=cycle_key,
            per_asset_signals=per_asset_signals,
            neutralized_signals=neutralized,
            fills=fills,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            dd_scale=dd_s,
            n_deployed_alphas=len(deployed_records),
            n_assets_traded=len([f for f in fills]),
            n_alphas_evaluated=n_evaluated,
        )

    def close(self) -> None:
        self.registry.close()
        self.portfolio_tracker.close()

    def reconcile(self) -> dict:
        results = {}
        for asset in self.tradeable_assets:
            ps = self.price_signals[asset]
            internal_qty = self.executor.get_position(ps)
            exchange_qty = self.executor.get_exchange_position(ps)
            internal_cash = self.executor.get_cash()
            exchange_cash = self.executor.get_exchange_cash()
            match = (
                abs(internal_qty - exchange_qty) < 1e-6
                and abs(internal_cash - exchange_cash) < 1.0
            )
            results[asset] = {
                "asset": asset,
                "internal_qty": internal_qty,
                "exchange_qty": exchange_qty,
                "match": match,
            }
        return results
