"""Evo daemon — continuous GP evolution writing candidates to SQLite queue."""
from __future__ import annotations

import gc
import hashlib
import logging
import resource
import signal
import time

import numpy as np

from ..alpha.evaluator import FAILED_FITNESS
from ..backtest.cost_model import CostModel
from ..backtest.engine import BacktestEngine
from ..config import Config, DATA_DIR
from ..data.universe import build_feature_list, price_signal
from ..dsl import to_string
from ..evolution.archive import AlphaArchive
from ..evolution.behavior import compute_behavior
from ..evolution.gp import GPConfig, GPEvolver

logger = logging.getLogger(__name__)


class EvoDaemon:
    """Continuous GP evolution daemon.

    Each round: load data → evolve (small pop, few generations) →
    write candidates to SQLite → gc → sleep → repeat.
    """

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config
        self.evo_cfg = config.evo_daemon
        self._running = False
        self._round = 0
        self._pop_size = self.evo_cfg.pop_size
        self.archive = AlphaArchive()

    def run(self) -> None:
        self._running = True
        self._setup_signals()
        logger.info(
            "EvoDaemon started: asset=%s, pop=%d, gens=%d, interval=%ds",
            self.asset, self._pop_size, self.evo_cfg.n_generations,
            self.evo_cfg.round_interval,
        )

        while self._running:
            try:
                self._run_round()
                self._round += 1
            except Exception:
                logger.exception("Round %d failed", self._round)
                self._sleep(60)
                continue

            self._check_memory()
            gc.collect()

            if self._running:
                self._sleep(self.evo_cfg.round_interval)

        logger.info("EvoDaemon stopped after %d rounds", self._round)

    def _run_round(self) -> None:
        t0 = time.perf_counter()

        # Load data
        features = build_feature_list(self.asset)
        data, prices, available_features = self._load_data(features)
        if data is None:
            logger.warning("Insufficient data, skipping round")
            return

        n_days = len(prices)

        # Build evaluator
        engine = BacktestEngine(
            CostModel(self.config.backtest.commission_pct,
                      self.config.backtest.slippage_pct)
        )

        def evaluate_fn(expr):
            try:
                sig = expr.evaluate(data)
                sig = np.asarray(sig, dtype=float)
                if sig.ndim == 0:
                    sig = np.full(n_days, float(sig))
                if len(sig) != n_days:
                    return FAILED_FITNESS
                if not np.all(np.isfinite(sig)):
                    sig = np.where(np.isfinite(sig), sig, 0.0)
                result = engine.run(sig, prices)
                v = result.fitness(self.config.fitness_metric)
                return v if np.isfinite(v) else FAILED_FITNESS
            except Exception:
                return FAILED_FITNESS

        # Evolve
        gp_cfg = GPConfig(
            pop_size=self._pop_size,
            n_generations=self.evo_cfg.n_generations,
            max_depth=self.config.generation.max_depth,
            bloat_penalty=self.config.generation.bloat_penalty,
            depth_penalty=self.config.generation.depth_penalty,
            similarity_penalty=self.config.generation.similarity_penalty,
        )
        seed = int(time.time()) ^ self._round
        evolver = GPEvolver(available_features, evaluate_fn, config=gp_cfg, seed=seed)
        results = evolver.run()

        # Fill archive
        live_signals: list[np.ndarray] = []
        for expr, fitness in results:
            try:
                sig = expr.evaluate(data)
                sig = np.nan_to_num(np.asarray(sig, dtype=float), nan=0.0)
                if sig.ndim == 0:
                    sig = np.full(n_days, float(sig))
                behavior = compute_behavior(sig, expr, live_signals)
                if self.archive.add(expr, fitness, behavior):
                    live_signals.append(sig)
            except Exception:
                continue

        # Filter positive fitness candidates
        candidates = [
            (expr, fitness) for expr, fitness in results
            if fitness > 0
        ][:self.evo_cfg.batch_size]

        # Write to candidates table
        if candidates:
            self._write_candidates(candidates)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Round %d: %d evolved, %d candidates written, "
            "archive %d/%d (%.1f%%), %.1fs",
            self._round, len(results), len(candidates),
            self.archive.size, self.archive.capacity,
            self.archive.coverage * 100, elapsed,
        )

    def _load_data(
        self, features: list[str],
    ) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None, list[str] | None]:
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
            from ..data.store import DataStore as DS
            store = DS(db_path, None)

        matrix = store.get_matrix(features)
        store.close()

        ps = price_signal(self.asset)
        if ps in matrix.columns:
            matrix = matrix[matrix[ps].notna()]
        matrix = matrix.bfill().fillna(0)

        if len(matrix) < self.config.backtest.min_days:
            return None, None, None

        available = [
            f for f in features
            if f in matrix.columns and not (matrix[f] == 0).all()
        ]
        data = {col: matrix[col].values for col in matrix.columns}
        prices = data[ps]
        return data, prices, available

    def _write_candidates(self, candidates: list[tuple]) -> None:
        import json
        import sqlite3

        from ..config import asset_data_dir
        db_path = asset_data_dir(self.asset) / "alpha_registry.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")

        now = time.time()
        rows = []
        for expr, fitness in candidates:
            expr_str = to_string(expr)
            cid = hashlib.md5(
                f"{expr_str}:{now}:{self._round}".encode(),
                usedforsecurity=False,
            ).hexdigest()[:16]
            rows.append((
                f"evo_{cid}", expr_str, fitness, "pending",
                json.dumps({}), now,
            ))

        conn.executemany(
            """INSERT OR IGNORE INTO candidates
            (candidate_id, expression, fitness, status, behavior_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        conn.close()

    def _check_memory(self) -> None:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_kb / 1024
        limit = self.evo_cfg.memory_limit_mb

        if rss_mb > limit:
            old_pop = self._pop_size
            self._pop_size = max(20, self._pop_size // 2)
            logger.warning(
                "RSS %.0fMB > limit %dMB, reducing pop_size %d → %d",
                rss_mb, limit, old_pop, self._pop_size,
            )

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
