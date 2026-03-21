"""Unified alpha generator — scores each expression across all assets."""
from __future__ import annotations

import gc
import logging
import random as _random
import signal
import time
from dataclasses import dataclass

import numpy as np

from alpha_os.alpha.cross_asset import evaluate_cross_asset_multi_horizon, DEFAULT_HORIZONS
from alpha_os.alpha.evaluator import FAILED_FITNESS, sanitize_signal
from alpha_os.backtest.benchmark import build_benchmark_returns
from alpha_os.config import Config, DATA_DIR, asset_data_dir
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.store import DataStore
from alpha_os.data.universe import (
    build_feature_list,
    init_universe,
    price_signal,
    stratified_feature_subset,
)
from alpha_os.dsl import to_string
from alpha_os.dsl.generator import AlphaGenerator
from alpha_os.evolution.behavior import compute_behavior
from alpha_os.evolution.discovery_pool import DiscoveryPool

logger = logging.getLogger(__name__)

# Minimum history for meaningful cross-asset evaluation
MIN_HISTORY_DAYS = 2000


@dataclass(frozen=True)
class AdmissionQueueCandidate:
    expression: str
    fitness: float
    queue_score: float
    behavior: np.ndarray
    best_horizon: int = 1


def _admission_queue_score(fitness: float) -> float:
    return max(fitness, 0.0)


def _survival_score(fitness: float, behavior) -> float:
    return fitness


class UnifiedAlphaGeneratorDaemon:
    """Cross-asset alpha generator.

    Evaluates each candidate expression against a diverse subset of assets
    (auto-selected via correlation clustering) and stores the mean residual
    fitness. One shared registry is used.
    """

    def __init__(self, config: Config):
        self.config = config
        self.generator_cfg = config.alpha_generator
        self.primary_asset = "BTC"
        self.universe: list[str] = []
        self.pool = DiscoveryPool()
        self._budget = self.generator_cfg.pop_size
        self._round = 0
        self._running = False
        self._client = build_signal_client_from_config(config.api)
        self._store = DataStore(DATA_DIR / "alpha_cache.db", self._client)

    def _sync_data(self, features: list[str]) -> None:
        """Ensure local cache has full history from signal-noise."""
        logger.info("Syncing %d features (min_history=%d days)...",
                     len(features), MIN_HISTORY_DAYS)
        self._store.sync(features, min_history_days=MIN_HISTORY_DAYS)

    def _load_data(
        self, features: list[str],
    ) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None, list[str]]:
        matrix = self._store.get_matrix(features)
        if matrix is None or len(matrix) < 2:
            return None, None, []
        data = {col: matrix[col].values for col in matrix.columns}
        ps = price_signal(self.primary_asset)
        prices = data.get(ps)
        if prices is None:
            return None, None, []
        finite_prices = prices[np.isfinite(prices)]
        if len(finite_prices) < 200:
            return None, None, []
        available = [
            f for f in features
            if f in matrix.columns and not (matrix[f] == 0).all()
        ]
        return data, prices, available

    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Initialize universe from API (not filesystem)
        universe_signals = init_universe(self._client)
        logger.info(
            "UnifiedAlphaGenerator started: %d universe signals, budget=%d",
            len(universe_signals), self._budget,
        )

        while self._running:
            try:
                self._run_round()
                self._round += 1
            except Exception:
                logger.exception("Round %d failed", self._round)
                self._sleep(60)
                continue

            gc.collect()
            if self._running:
                self._sleep(self.generator_cfg.round_interval)

        logger.info("UnifiedAlphaGenerator stopped after %d rounds", self._round)

    def _run_round(self) -> None:
        t0 = time.perf_counter()

        all_features = build_feature_list(self.primary_asset, self._client)
        k = self.generator_cfg.feature_subset_k
        seed = int(time.time()) ^ self._round

        subset = stratified_feature_subset(all_features, k=k, seed=seed)
        ps = price_signal(self.primary_asset)

        # Load features: price + subset + universe candidates for eval
        universe_signals = init_universe(self._client) if not self.universe else self.universe
        load_features = sorted({ps} | subset | set(universe_signals))

        # Sync from signal-noise (backfills short history automatically)
        if self._round == 0:
            self._sync_data(load_features)

        data, prices, available_features = self._load_data(load_features)
        if data is None:
            logger.warning("Insufficient data, skipping round")
            return

        # Use cached eval universe; recompute only if missing
        if not self.universe:
            from alpha_os.data.eval_universe import (
                load_cached_eval_universe,
                save_eval_universe,
                select_eval_universe,
            )
            self.universe = load_cached_eval_universe()
            if not self.universe:
                self.universe = select_eval_universe(
                    data, universe_signals,
                    n_clusters=20, min_finite_days=500,
                )
                if self.universe:
                    save_eval_universe(self.universe)
            if not self.universe:
                logger.warning("No assets with sufficient data for evaluation")
                return

        # Build benchmark
        bm_assets = self.config.backtest.benchmark_assets
        bm_returns = None
        if bm_assets:
            bm_returns = build_benchmark_returns(data, bm_assets)
            if len(bm_returns) == 0:
                bm_returns = None

        n_days = len(prices)
        rng = _random.Random(seed)

        available_subset = frozenset(available_features) - set(self.universe)
        generator = AlphaGenerator(
            available_features,
            feature_subset=available_subset,
            seed=seed,
        )

        # Generate candidates
        budget = self._budget
        candidates = []

        if self.pool.size > 0:
            n_mutate = int(budget * self.generator_cfg.mutate_ratio)
            elites = self.pool.sample(n_mutate, rng=rng)
            for entry in elites:
                candidates.append(generator.mutate(entry.expr))

        n_random = budget - len(candidates)
        candidates.extend(generator.generate_random(
            n_random, max_depth=self.config.generation.max_depth,
        ))

        # Evaluate each candidate across all assets
        n_stored = 0
        n_replaced = 0
        queued_candidates: list[AdmissionQueueCandidate] = []

        for expr in candidates:
            try:
                expr_str = to_string(expr)
                result = evaluate_cross_asset_multi_horizon(
                    expr_str, data, self.universe,
                    horizons=DEFAULT_HORIZONS,
                    fitness_metric="ic",  # fixed: IC for signal evaluation
                    benchmark_assets=bm_assets,
                )
                if not result.per_asset:
                    continue

                fitness = result.best_fitness
                best_horizon = result.best_horizon
                if not np.isfinite(fitness) or fitness <= FAILED_FITNESS:
                    continue

                sig = sanitize_signal(expr.evaluate(data))
                if sig.ndim == 0:
                    sig = np.full(n_days, float(sig))
                behavior = compute_behavior(sig, expr, prices=prices)

                update = self.pool.store_candidate(
                    expr, behavior, sig,
                    fitness=fitness,
                    survival_score=_survival_score(fitness, behavior),
                    best_horizon=best_horizon,
                )
                if update.stored:
                    n_stored += 1
                    if update.replaced:
                        n_replaced += 1

                if fitness >= self.generator_cfg.promotion_min_fitness:
                    queued_candidates.append(AdmissionQueueCandidate(
                        expression=expr_str,
                        fitness=fitness,
                        queue_score=_admission_queue_score(fitness),
                        behavior=behavior,
                        best_horizon=best_horizon,
                    ))
            except Exception:
                continue

        if queued_candidates:
            self._enqueue_candidates(queued_candidates)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Round %d: %d candidates, %d stored (%d replaced), "
            "%d queued, %.1fs, pool=%d, universe=%d assets",
            self._round, len(candidates), n_stored, n_replaced,
            len(queued_candidates), elapsed, self.pool.size,
            len(self.universe),
        )

    def _enqueue_candidates(self, candidates: list[AdmissionQueueCandidate]) -> int:
        from alpha_os.alpha.admission_queue import CandidateSeed
        from alpha_os.alpha.managed_alphas import ManagedAlphaStore

        limit = self.generator_cfg.promote_per_round
        if limit <= 0:
            return 0
        candidates.sort(key=lambda c: c.queue_score, reverse=True)
        candidates = candidates[:limit]

        adir = asset_data_dir(self.primary_asset)
        store = ManagedAlphaStore(adir / "alpha_registry.db")
        try:
            seeds = [
                CandidateSeed(
                    expression=c.expression,
                    source="unified_generator",
                    fitness=c.fitness,
                    behavior_json={
                        "best_horizon": c.best_horizon,
                        "behavior": [float(x) for x in c.behavior.tolist()],
                    },
                )
                for c in candidates
            ]
            n = store.queue_candidates(seeds)
            logger.info("Enqueued %d candidates into registry", n)
            return n
        finally:
            store.close()

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        self._running = False

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while self._running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))
