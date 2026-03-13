"""Alpha generator daemon — continuous GP evolution into the archive."""
from __future__ import annotations

import gc
import logging
import resource
import signal
import time
from dataclasses import dataclass

import numpy as np

from ..alpha.evaluator import FAILED_FITNESS, sanitize_signal
from ..alpha.managed_alphas import CandidateSeed, ManagedAlphaStore
from ..backtest.cost_model import CostModel
from ..backtest.engine import BacktestEngine
from ..config import Config, DATA_DIR, asset_data_dir
from ..dsl import to_string
from ..data.universe import build_feature_list, price_signal
from ..dsl.generator import AlphaGenerator
from ..evolution.discovery_pool import DiscoveryPool
from ..evolution.behavior import compute_behavior
from ..evolution.gp import GPConfig, GPEvolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionCandidate:
    expression: str
    fitness: float
    behavior: np.ndarray


class AlphaGeneratorDaemon:
    """Continuous GP evolution daemon.

    Each round: load data → evolve on a stratified feature subset →
    archive new diverse candidates → gc → sleep → repeat.
    """

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config
        self.generator_cfg = config.alpha_generator
        self._running = False
        self._round = 0
        self._pop_size = self.generator_cfg.pop_size

        db_path = asset_data_dir(asset) / "archive.db"
        self.archive = DiscoveryPool.load_from_db(db_path)
        logger.info("Loaded MAP-Elites archive: %d entries", self.archive.size)

    def run(self) -> None:
        self._running = True
        self._setup_signals()
        logger.info(
            "AlphaGeneratorDaemon started: asset=%s, mode=map_elites, pop=%d, gens=%d, interval=%ds",
            self.asset, self._pop_size,
            self.generator_cfg.n_generations, self.generator_cfg.round_interval,
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
                self._sleep(self.generator_cfg.round_interval)

        logger.info("AlphaGeneratorDaemon stopped after %d rounds", self._round)

    def _run_round(self) -> None:
        """MAP-Elites round: stratified feature subset → GP → sanity filter → archive."""
        t0 = time.perf_counter()

        features = build_feature_list(self.asset)
        data, prices, available_features = self._load_data(features)
        if data is None:
            logger.warning("Insufficient data, skipping round")
            return

        n_days = len(prices)
        k = self.generator_cfg.feature_subset_k
        seed = int(time.time()) ^ self._round

        # Random feature subset for this round
        generator = AlphaGenerator.with_stratified_subset(available_features, k=k, seed=seed)
        subset = generator.feature_subset

        # Build evaluator (still uses fitness to guide GP search)
        engine = BacktestEngine(
            CostModel(self.config.backtest.commission_pct,
                      self.config.backtest.slippage_pct),
            allow_short=self.config.trading.supports_short,
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

        # Evolve with feature subset
        gp_cfg = GPConfig(
            pop_size=self._pop_size,
            n_generations=self.generator_cfg.n_generations,
            max_depth=self.config.generation.max_depth,
            bloat_penalty=self.config.generation.bloat_penalty,
        )
        evolver = GPEvolver(
            available_features, evaluate_fn, config=gp_cfg,
            seed=seed, generator=generator,
        )
        results = evolver.run()

        # Fill discovery pool with sanity filter (no fitness competition)
        n_added = 0
        promoted: list[PromotionCandidate] = []
        for expr, _fitness in results:
            try:
                sig = expr.evaluate(data)
                sig = sanitize_signal(sig)
                if sig.ndim == 0:
                    sig = np.full(n_days, float(sig))
                behavior = compute_behavior(sig, expr, feature_subset=subset)
                if self.archive.add_if_empty(expr, behavior, sig):
                    n_added += 1
                    promoted.append(
                        PromotionCandidate(
                            expression=to_string(expr),
                            fitness=float(_fitness),
                            behavior=behavior,
                        )
                    )
            except Exception:
                continue

        # Persist discovery pool
        db_path = asset_data_dir(self.asset) / "archive.db"
        if n_added > 0:
            self.archive.save_to_db(db_path)
        n_queued = self._queue_promoted_candidates(promoted)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Round %d [MAP-Elites]: %d evolved, %d added, %d queued, "
            "discovery_pool %d/%d (%.1f%%), subset=%d features, %.1fs",
            self._round, len(results), n_added, n_queued,
            self.archive.size, self.archive.capacity,
            self.archive.coverage * 100, len(subset) if subset else 0, elapsed,
        )

    def _queue_promoted_candidates(
        self,
        candidates: list[PromotionCandidate],
    ) -> int:
        if not candidates:
            return 0

        limit = self.generator_cfg.promote_per_round
        if limit <= 0:
            return 0

        promoted = [
            candidate
            for candidate in candidates
            if candidate.fitness >= self.generator_cfg.promotion_min_fitness
        ]
        if not promoted:
            return 0

        promoted.sort(key=lambda candidate: candidate.fitness, reverse=True)
        promoted = promoted[:limit]
        seeds = [
            CandidateSeed(
                expression=candidate.expression,
                source=f"alpha_generator_{self.asset.lower()}",
                fitness=candidate.fitness,
                behavior_json={
                    "source": "alpha_generator",
                    "asset": self.asset,
                    "behavior": [float(x) for x in candidate.behavior.tolist()],
                    "round": self._round,
                },
            )
            for candidate in promoted
        ]
        store = ManagedAlphaStore(asset_data_dir(self.asset) / "alpha_registry.db")
        try:
            return store.queue_candidates(seeds)
        finally:
            store.close()

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

    def _check_memory(self) -> None:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_kb / 1024
        limit = self.generator_cfg.memory_limit_mb

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
