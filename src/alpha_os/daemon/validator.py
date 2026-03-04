"""Validator daemon — poll candidates queue, validate, adopt to registry."""
from __future__ import annotations

import logging
import signal
import sqlite3
import time

import numpy as np

from ..alpha.evaluator import FAILED_FITNESS, EvaluationError, evaluate_expression, normalize_signal
from ..alpha.registry import AlphaRecord, AlphaRegistry, AlphaState
from ..backtest.cost_model import CostModel
from ..backtest.engine import BacktestEngine
from ..config import Config, DATA_DIR, asset_data_dir
from ..data.universe import build_feature_list, price_signal
from ..dsl import parse, to_string
from ..governance.gates import GateConfig, adoption_gate
from ..validation.deflated_sharpe import deflated_sharpe_ratio
from ..validation.pbo import probability_of_backtest_overfitting
from ..validation.purged_cv import purged_walk_forward

logger = logging.getLogger(__name__)


class ValidatorDaemon:
    """Poll candidates table, validate, and adopt passing alphas."""

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config
        self.val_cfg = config.validator
        self._running = False
        self._round = 0

    def run(self) -> None:
        self._running = True
        self._setup_signals()
        logger.info(
            "ValidatorDaemon started: asset=%s, poll=%ds, batch=%d, min_queue=%d",
            self.asset, self.val_cfg.poll_interval,
            self.val_cfg.batch_size, self.val_cfg.min_queue_size,
        )

        while self._running:
            try:
                n_pending = self._count_pending()
                if n_pending >= self.val_cfg.min_queue_size:
                    self._run_batch()
                    self._round += 1
                    self._gc_old_candidates()
                else:
                    logger.debug("Queue: %d pending (< %d), sleeping",
                                 n_pending, self.val_cfg.min_queue_size)
            except Exception:
                logger.exception("Validation round %d failed", self._round)

            if self._running:
                self._sleep(self.val_cfg.poll_interval)

        logger.info("ValidatorDaemon stopped after %d rounds", self._round)

    def _count_pending(self) -> int:
        conn = self._open_registry_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE status = 'pending'"
        ).fetchone()
        conn.close()
        return row[0]

    def _run_batch(self) -> None:
        t0 = time.perf_counter()

        # Fetch pending candidates
        conn = self._open_registry_conn()
        rows = conn.execute(
            "SELECT candidate_id, expression, fitness FROM candidates "
            "WHERE status = 'pending' ORDER BY fitness DESC LIMIT ?",
            (self.val_cfg.batch_size,),
        ).fetchall()
        conn.close()

        if not rows:
            return

        # Mark as validating
        cids = [r[0] for r in rows]
        conn = self._open_registry_conn()
        conn.executemany(
            "UPDATE candidates SET status = 'validating' WHERE candidate_id = ?",
            [(cid,) for cid in cids],
        )
        conn.commit()
        conn.close()

        # Load data
        features = build_feature_list(self.asset)
        data, prices, n_days = self._load_data(features)
        if data is None:
            logger.warning("Insufficient data for validation, skipping batch")
            self._reset_to_pending(cids)
            return

        engine = BacktestEngine(
            CostModel(self.config.backtest.commission_pct,
                      self.config.backtest.slippage_pct)
        )

        # Parse and evaluate candidates
        parsed = []
        for cid, expr_str, fitness in rows:
            try:
                expr = parse(expr_str)
                parsed.append((cid, expr, expr_str, fitness))
            except Exception:
                self._reject_candidate(cid, "parse error")

        # Validate: purged WF-CV + DSR
        n_trials = len(parsed)
        validated = []
        n_failed = 0

        for cid, expr, expr_str, fitness in parsed:
            try:
                sig = evaluate_expression(expr, data, n_days)

                cv = purged_walk_forward(
                    sig, prices, engine,
                    n_folds=self.config.validation.n_cv_folds,
                    embargo=self.config.validation.embargo_days,
                )
                _metric = self.config.fitness_metric
                _oos_fit = cv.oos_fitness(_metric)
                if _oos_fit <= 0:
                    self._reject_candidate(cid, f"oos_{_metric}={_oos_fit:.3f}")
                    continue

                pos = normalize_signal(sig)
                rets = np.diff(prices) / prices[:-1]
                n = min(len(pos) - 1, len(rets))
                strat_rets = pos[:n] * rets[:n]
                dsr = deflated_sharpe_ratio(strat_rets, n_trials=max(n_trials, 1))

                validated.append((cid, expr, expr_str, fitness, cv, dsr.p_value))
            except (EvaluationError, Exception) as exc:
                n_failed += 1
                self._reject_candidate(cid, str(exc)[:200])

        if n_failed:
            logger.info("Validation: %d candidates failed evaluation", n_failed)

        if not validated:
            logger.info("Batch: 0 candidates passed validation")
            return

        # Batch PBO
        batch_pbo = self._compute_batch_pbo(validated, data, prices, engine)

        # Adoption gate
        gate_cfg = GateConfig(
            oos_sharpe_min=self.config.validation.oos_sharpe_min,
            pbo_max=self.config.validation.pbo_max,
            dsr_pvalue_max=self.config.validation.dsr_pvalue_max,
        )

        registry = AlphaRegistry(asset_data_dir(self.asset) / "alpha_registry.db")
        n_adopted = 0
        n_rejected = 0

        for cid, expr, expr_str, fitness, cv, dsr_pvalue in validated:
            result = adoption_gate(
                oos_sharpe=cv.oos_sharpe,
                oos_log_growth=cv.oos_expected_log_growth,
                oos_cvar_95=cv.oos_cvar_95,
                oos_tail_hit_rate=cv.oos_tail_hit_rate,
                pbo=batch_pbo,
                dsr_pvalue=dsr_pvalue,
                fdr_passed=True,
                avg_correlation=0.0,
                n_days=n_days,
                config=gate_cfg,
            )

            if result.passed:
                alpha_id = f"v2_{hash(expr_str) % 10**8:08d}"
                record = AlphaRecord(
                    alpha_id=alpha_id,
                    expression=expr_str,
                    state=AlphaState.ACTIVE,
                    fitness=fitness,
                    oos_sharpe=cv.oos_sharpe,
                    oos_log_growth=cv.oos_expected_log_growth,
                    pbo=batch_pbo,
                    dsr_pvalue=dsr_pvalue,
                )
                registry.register(record)
                self._adopt_candidate(cid, cv.oos_sharpe, batch_pbo, dsr_pvalue)

                # Write diversity cache
                self._write_diversity_cache(alpha_id, registry)
                n_adopted += 1
            else:
                reason = "; ".join(result.reasons[:3])
                self._reject_candidate(cid, reason)
                n_rejected += 1

        registry.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Round %d: %d candidates → %d validated → %d adopted, %d rejected (%.1fs)",
            self._round, len(rows), len(validated), n_adopted, n_rejected, elapsed,
        )

    def _compute_batch_pbo(self, validated, data, prices, engine) -> float:
        n_days = len(prices)
        signals = []
        for cid, expr, expr_str, fitness, cv, dsr_pvalue in validated:
            try:
                sig = evaluate_expression(expr, data, n_days)
                signals.append(sig)
            except EvaluationError:
                continue

        if len(signals) < 2:
            return 1.0

        if len(signals) > 200:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(signals), 200, replace=False)
            signals = [signals[i] for i in indices]

        sig_matrix = np.array(signals)
        try:
            pbo_result = probability_of_backtest_overfitting(
                sig_matrix, prices, engine,
                n_blocks=10, max_combinations=50,
            )
            logger.info("Batch PBO: %.3f (%d strategies)", pbo_result.pbo, len(signals))
            return pbo_result.pbo
        except Exception:
            logger.warning("PBO computation failed, using 1.0")
            return 1.0

    def _write_diversity_cache(self, alpha_id: str, registry: AlphaRegistry) -> None:
        n_active = registry.count(AlphaState.ACTIVE)
        conn = self._open_registry_conn()
        conn.execute(
            "INSERT OR REPLACE INTO diversity_cache "
            "(alpha_id, diversity_score, computed_at, n_alphas_compared) "
            "VALUES (?, ?, ?, ?)",
            (alpha_id, 1.0, time.time(), n_active),
        )
        conn.commit()
        conn.close()

    def _load_data(self, features):
        from ..data.store import DataStore

        db_path = DATA_DIR / "alpha_cache.db"
        try:
            from signal_noise.client import SignalClient
            client = SignalClient(
                base_url=self.config.api.base_url,
                timeout=self.config.api.timeout,
            )
            store = DataStore(db_path, client)
            try:
                if client.health():
                    store.sync(features)
            except Exception as exc:
                logger.warning("API sync failed: %s", exc)
        except ImportError:
            store = DataStore(db_path, None)

        matrix = store.get_matrix(features)
        store.close()

        ps = price_signal(self.asset)
        if ps in matrix.columns:
            matrix = matrix[matrix[ps].notna()]
        matrix = matrix.bfill().fillna(0)

        if len(matrix) < self.config.backtest.min_days:
            return None, None, None

        data = {col: matrix[col].values for col in matrix.columns}
        prices = data[ps]
        return data, prices, len(prices)

    def _open_registry_conn(self) -> sqlite3.Connection:
        db_path = asset_data_dir(self.asset) / "alpha_registry.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _reject_candidate(self, cid: str, reason: str) -> None:
        conn = self._open_registry_conn()
        conn.execute(
            "UPDATE candidates SET status = 'rejected', "
            "validated_at = ?, error_message = ? WHERE candidate_id = ?",
            (time.time(), reason[:200], cid),
        )
        conn.commit()
        conn.close()

    def _adopt_candidate(self, cid: str, oos_sharpe: float, pbo: float, dsr: float) -> None:
        conn = self._open_registry_conn()
        conn.execute(
            "UPDATE candidates SET status = 'adopted', "
            "oos_sharpe = ?, pbo = ?, dsr_pvalue = ?, validated_at = ? "
            "WHERE candidate_id = ?",
            (oos_sharpe, pbo, dsr, time.time(), cid),
        )
        conn.commit()
        conn.close()

    def _reset_to_pending(self, cids: list[str]) -> None:
        conn = self._open_registry_conn()
        conn.executemany(
            "UPDATE candidates SET status = 'pending' WHERE candidate_id = ?",
            [(cid,) for cid in cids],
        )
        conn.commit()
        conn.close()

    def _gc_old_candidates(self, max_age_days: int = 30) -> None:
        cutoff = time.time() - max_age_days * 86400
        conn = self._open_registry_conn()
        result = conn.execute(
            "DELETE FROM candidates WHERE status IN ('adopted', 'rejected') "
            "AND created_at < ?",
            (cutoff,),
        )
        if result.rowcount > 0:
            logger.info("GC: deleted %d old candidates", result.rowcount)
        conn.commit()
        conn.close()

    def _sleep(self, seconds: int) -> None:
        end = time.time() + seconds
        while self._running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    def _setup_signals(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (OSError, ValueError):
            pass

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        self._running = False
