"""Unified alpha generator — scores each expression across all assets."""
from __future__ import annotations

import gc
import logging
import random as _random
import signal
import time

import numpy as np

from alpha_os.alpha.cross_asset import evaluate_cross_asset
from alpha_os.alpha.evaluator import FAILED_FITNESS, sanitize_signal
from alpha_os.daemon.alpha_generator import AdmissionQueueCandidate
from alpha_os.backtest.benchmark import build_benchmark_returns
from alpha_os.config import Config, DATA_DIR, asset_data_dir
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.store import DataStore
from alpha_os.data.universe import (
    CROSS_ASSET_UNIVERSE,
    build_feature_list,
    price_signal,
    stratified_feature_subset,
)
from alpha_os.dsl import to_string
from alpha_os.dsl.generator import AlphaGenerator
from alpha_os.evolution.behavior import compute_behavior
from alpha_os.evolution.discovery_pool import DiscoveryPool

logger = logging.getLogger(__name__)


def _admission_queue_score(fitness: float) -> float:
    return max(fitness, 0.0)


def _survival_score(fitness: float, behavior) -> float:
    return fitness


class UnifiedAlphaGeneratorDaemon:
    """Cross-asset alpha generator.

    Unlike the per-asset AlphaGeneratorDaemon, this evaluates each
    candidate expression against ALL assets in CROSS_ASSET_UNIVERSE
    and stores the mean residual fitness. One shared registry is used.
    """

    def __init__(self, config: Config):
        self.config = config
        self.generator_cfg = config.alpha_generator

        # Use first crypto asset as primary for data loading & behavior
        self.primary_asset = "BTC"
        self.universe = CROSS_ASSET_UNIVERSE

        self.archive = DiscoveryPool()

        self._budget = self.generator_cfg.pop_size
        self._round = 0
        self._running = False

    def _load_data(
        self, features: list[str],
    ) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None, list[str]]:
        client = build_signal_client_from_config(self.config.api)
        store = DataStore(DATA_DIR / "alpha_cache.db", client)
        matrix = store.get_matrix(features)
        if matrix is None or len(matrix) < 200:
            return None, None, []
        available = [
            f for f in features
            if f in matrix.columns and not (matrix[f] == 0).all()
        ]
        data = {col: matrix[col].values for col in matrix.columns}
        ps = price_signal(self.primary_asset)
        prices = data.get(ps)
        if prices is None or len(prices) < 200:
            return None, None, []
        return data, prices, available

    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info(
            "UnifiedAlphaGenerator started: universe=%d assets, budget=%d, interval=%ds",
            len(self.universe), self._budget, self.generator_cfg.round_interval,
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

        # Load features — use full catalog, not asset-specific
        all_features = build_feature_list(self.primary_asset)
        k = self.generator_cfg.feature_subset_k
        seed = int(time.time()) ^ self._round

        subset = stratified_feature_subset(all_features, k=k, seed=seed)
        # Include ALL universe price signals for cross-asset evaluation
        universe_signals = set()
        for asset_sig in self.universe:
            universe_signals.add(asset_sig)
        load_features = sorted(universe_signals | subset)

        data, prices, available_features = self._load_data(load_features)
        if data is None:
            logger.warning("Insufficient data, skipping round")
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

        if self.archive.size > 0:
            n_mutate = int(budget * self.generator_cfg.mutate_ratio)
            elites = self.archive.sample(n_mutate, rng=rng)
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

                # Cross-asset fitness: mean residual fitness across universe
                per_asset = evaluate_cross_asset(
                    expr_str,
                    data,
                    self.universe,
                    fitness_metric=self.config.fitness_metric,
                    commission_pct=self.config.backtest.commission_pct,
                    slippage_pct=self.config.backtest.slippage_pct,
                    allow_short=self.config.trading.supports_short,
                    benchmark_assets=bm_assets,
                )

                if not per_asset:
                    continue

                fitness = float(np.mean(list(per_asset.values())))

                if not np.isfinite(fitness) or fitness <= FAILED_FITNESS:
                    continue

                # Behavior computed on primary asset
                sig = sanitize_signal(expr.evaluate(data))
                if sig.ndim == 0:
                    sig = np.full(n_days, float(sig))
                behavior = compute_behavior(sig, expr, prices=prices)

                update = self.archive.store_candidate(
                    expr,
                    behavior,
                    sig,
                    fitness=fitness,
                    survival_score=_survival_score(fitness, behavior),
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
                    ))

            except Exception:
                continue

        if queued_candidates:
            self._enqueue_candidates(queued_candidates)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Round %d: %d candidates, %d stored (%d replaced), "
            "%d queued, %.1fs, archive=%d, universe=%d assets",
            self._round, len(candidates), n_stored, n_replaced,
            len(queued_candidates), elapsed, self.archive.size,
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
                )
                for c in candidates
            ]
            n = store.enqueue_candidates(seeds)
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
