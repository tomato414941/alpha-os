"""Admission daemon — poll candidates queue, validate via OOS IC, adopt to registry."""
from __future__ import annotations

import logging
import signal
import sqlite3
import time

import numpy as np

from ..alpha.admission_queue import (
    adopt_candidate,
    count_pending_candidates,
    fetch_pending_candidates,
    gc_old_candidate_results,
    mark_candidates_validating,
    reject_candidate,
    reset_candidates_to_pending,
)
from ..alpha.admission_replay import alpha_id_for_expression
from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..alpha.managed_alphas import AlphaRecord, ManagedAlphaStore, AlphaState
from ..config import Config, SIGNAL_CACHE_DB, asset_data_dir
from ..data.signal_client import build_signal_client_from_config
from ..data.universe import build_feature_list, price_signal
from ..dsl import parse
from ..validation.deflated_sharpe import deflated_sharpe_ratio
from ..validation.purged_cv import purged_walk_forward_ic

logger = logging.getLogger(__name__)


class AdmissionDaemon:
    """Poll candidates table, validate via OOS IC, and adopt passing alphas."""

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config
        self.admission_cfg = config.admission
        self._running = False
        self._round = 0

    def run(self) -> None:
        self._running = True
        self._setup_signals()
        logger.info(
            "AdmissionDaemon started: asset=%s, poll=%ds, batch=%d, min_queue=%d",
            self.asset, self.admission_cfg.poll_interval,
            self.admission_cfg.batch_size, self.admission_cfg.min_queue_size,
        )

        while self._running:
            try:
                n_pending = self._count_pending()
                if n_pending >= self.admission_cfg.min_queue_size:
                    self._run_batch()
                    self._round += 1
                    self._gc_old_candidates()
                else:
                    logger.debug("Queue: %d pending (< %d), sleeping",
                                 n_pending, self.admission_cfg.min_queue_size)
            except Exception:
                logger.exception("Admission round %d failed", self._round)

            if self._running:
                self._sleep(self.admission_cfg.poll_interval)

        logger.info("AdmissionDaemon stopped after %d rounds", self._round)

    def _count_pending(self) -> int:
        conn = self._open_registry_conn()
        try:
            return count_pending_candidates(conn)
        finally:
            conn.close()

    def _fetch_pending_rows(self, limit: int) -> list[tuple[str, str, float, str]]:
        conn = self._open_registry_conn()
        try:
            return fetch_pending_candidates(conn, limit)
        finally:
            conn.close()

    def _run_batch(self) -> None:
        t0 = time.perf_counter()

        # Fetch pending candidates
        rows = self._fetch_pending_rows(self.admission_cfg.batch_size)

        if not rows:
            return

        # Mark as validating
        cids = [r[0] for r in rows]
        conn = self._open_registry_conn()
        try:
            mark_candidates_validating(conn, cids)
        finally:
            conn.close()

        # Load data
        features = build_feature_list(self.asset)
        data, prices, n_days = self._load_data(features)
        if data is None:
            logger.warning("Insufficient data for validation, skipping batch")
            self._reset_to_pending(cids)
            return

        # Build benchmark for residual IC
        bm_returns = None
        bm_assets = self.config.backtest.benchmark_assets
        if bm_assets:
            from alpha_os.backtest.benchmark import build_benchmark_returns
            bm_returns = build_benchmark_returns(data, bm_assets)
            if len(bm_returns) == 0:
                bm_returns = None

        # Parse and evaluate candidates
        import json as _json
        parsed = []
        for cid, expr_str, fitness, behavior_json_str in rows:
            try:
                expr = parse(expr_str)
                try:
                    bj = _json.loads(behavior_json_str) if behavior_json_str else {}
                except (ValueError, TypeError):
                    bj = {}
                horizon = int(bj.get("best_horizon", 1))
                parsed.append((cid, expr, expr_str, fitness, horizon))
            except Exception:
                self._reject_candidate(cid, "parse error")

        # Validate: purged WF-CV (IC-based) + DSR
        n_trials = len(parsed)
        validated = []
        n_failed = 0

        for cid, expr, expr_str, fitness, horizon in parsed:
            try:
                sig = evaluate_expression(expr, data, n_days)

                cv = purged_walk_forward_ic(
                    sig, prices,
                    horizon=horizon,
                    n_folds=self.config.validation.n_cv_folds,
                    embargo=self.config.validation.embargo_days,
                    benchmark_returns=bm_returns,
                )
                if cv.oos_ic <= 0:
                    self._reject_candidate(cid, f"oos_ic={cv.oos_ic:.4f}")
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

        registry = ManagedAlphaStore(asset_data_dir(self.asset) / "alpha_registry.db")
        existing_ids = {
            record.expression: record.alpha_id
            for record in registry.list_all()
        }
        n_adopted = 0
        n_rejected = 0
        objective = self.config.portfolio.objective

        for cid, expr, expr_str, fitness, cv, dsr_pvalue in validated:
            alpha_id = alpha_id_for_expression(
                expr_str,
                existing_ids=existing_ids,
            )
            existing_ids[expr_str] = alpha_id
            record = AlphaRecord(
                alpha_id=alpha_id,
                expression=expr_str,
                state=AlphaState.ACTIVE,
                fitness=fitness,
                oos_sharpe=cv.oos_sharpe,
                oos_log_growth=cv.oos_expected_log_growth,
                pbo=0.0,
                dsr_pvalue=dsr_pvalue,
                stake=max(cv.oos_fitness(objective), 0.0),
            )
            registry.register(record)
            self._adopt_candidate(cid, cv.oos_sharpe, 0.0, dsr_pvalue)
            n_adopted += 1

        registry.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Round %d: %d candidates -> %d validated -> %d adopted, %d rejected (%.1fs)",
            self._round, len(rows), len(validated), n_adopted, n_rejected, elapsed,
        )

    def _load_data(self, features):
        from ..data.store import DataStore

        db_path = SIGNAL_CACHE_DB
        try:
            client = build_signal_client_from_config(self.config.api)
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
        matrix = matrix.fillna(0)

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
        try:
            reject_candidate(conn, cid, reason)
        finally:
            conn.close()

    def _adopt_candidate(self, cid: str, oos_sharpe: float, pbo: float, dsr: float) -> None:
        conn = self._open_registry_conn()
        try:
            adopt_candidate(
                conn,
                cid,
                oos_sharpe=oos_sharpe,
                pbo=pbo,
                dsr_pvalue=dsr,
            )
        finally:
            conn.close()

    def _reset_to_pending(self, cids: list[str]) -> None:
        conn = self._open_registry_conn()
        try:
            reset_candidates_to_pending(conn, cids)
        finally:
            conn.close()

    def _gc_old_candidates(self, max_age_days: int = 30) -> None:
        conn = self._open_registry_conn()
        try:
            deleted = gc_old_candidate_results(conn, max_age_days=max_age_days)
        finally:
            conn.close()
        if deleted > 0:
            logger.info("GC: deleted %d old candidates", deleted)

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
