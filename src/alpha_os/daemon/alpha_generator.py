"""Alpha generator daemon — pure MAP-Elites candidate generation."""
from __future__ import annotations

import gc
import logging
import random as _random
import resource
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..alpha.evaluator import FAILED_FITNESS, sanitize_signal
from ..alpha.managed_alphas import CandidateSeed, ManagedAlphaStore
from ..alpha.expression_identity import expression_semantic_key
from ..backtest.cost_model import CostModel
from ..backtest.engine import BacktestEngine
from ..config import Config, DATA_DIR, asset_data_dir
from ..data.signal_client import build_signal_client_from_config
from ..dsl import parse, to_string
from ..data.universe import build_feature_list, price_signal, stratified_feature_subset
from ..dsl.generator import AlphaGenerator
from ..evolution.discovery_pool import DiscoveryPool
from ..evolution.behavior import compute_behavior

logger = logging.getLogger(__name__)

_MIN_BUDGET = 20


@dataclass(frozen=True)
class AdmissionQueueCandidate:
    expression: str
    fitness: float
    queue_score: float
    behavior: np.ndarray


def _survival_score(fitness: float, behavior: np.ndarray) -> float:
    """Cell-local survival score for discovery-pool incumbents.

    Within a behavior cell, novelty is already constrained by the descriptor
    itself. Bias survival slightly toward simpler expressions so that near-tied
    candidates do not drift toward unnecessary complexity.
    """
    complexity = float(behavior[2]) if len(behavior) >= 3 else 0.0
    simplicity_bonus = 1.0 / (1.0 + max(complexity, 0.0))
    return float(fitness) + 0.01 * simplicity_bonus


def _admission_queue_score(fitness: float) -> float:
    """Ranking score for admission-queue enqueue order."""
    return float(fitness)


def _existing_enqueue_semantic_keys(store: ManagedAlphaStore) -> set[str]:
    managed_keys = {
        expression_semantic_key(record.expression)
        for record in store.list_all()
        if record.state != "rejected"
    }
    queued_keys = {
        expression_semantic_key(expression)
        for expression in store.list_candidate_expressions(
            statuses=("pending", "validating", "adopted")
        )
    }
    return managed_keys | queued_keys


def _current_rss_mb() -> float:
    """Return current resident set size in MB."""
    status_path = Path("/proc/self/status")
    try:
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    except OSError:
        pass

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kb / 1024.0


def _load_generator_data(
    asset: str,
    config: Config,
    features: list[str],
) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None, list[str] | None]:
    from ..data.store import DataStore

    db_path = DATA_DIR / "alpha_cache.db"
    try:
        client = build_signal_client_from_config(config.api)
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

    ps = price_signal(asset)
    if ps in matrix.columns:
        matrix = matrix[matrix[ps].notna()]
    matrix = matrix.bfill().fillna(0)

    if len(matrix) < config.backtest.min_days:
        return None, None, None

    available = [
        f for f in features
        if f in matrix.columns and not (matrix[f] == 0).all()
    ]
    data = {col: matrix[col].values for col in matrix.columns}
    prices = data[ps]
    return data, prices, available


def _score_expression(
    expression: str,
    data: dict[str, np.ndarray],
    prices: np.ndarray,
    config: Config,
) -> float:
    n_days = len(prices)
    engine = BacktestEngine(
        CostModel(
            config.backtest.commission_pct,
            config.backtest.slippage_pct,
        ),
        allow_short=config.trading.supports_short,
    )
    try:
        expr = parse(expression)
        sig = sanitize_signal(expr.evaluate(data))
        if sig.ndim == 0:
            sig = np.full(n_days, float(sig))
        if len(sig) != n_days:
            return FAILED_FITNESS
        result = engine.run(sig, prices)
        value = result.fitness(config.fitness_metric)
        return value if np.isfinite(value) else FAILED_FITNESS
    except Exception:
        return FAILED_FITNESS


def enqueue_discovery_pool_candidates(
    asset: str,
    config: Config,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Queue top discovery-pool entries into candidates.

    Returns a tuple of `(selected_count, inserted_count)`.
    """
    pool = DiscoveryPool.load_from_db(asset_data_dir(asset) / "archive.db")
    promote_limit = config.alpha_generator.promote_per_round if limit is None else limit
    if promote_limit <= 0:
        return 0, 0

    enqueued = [
        AdmissionQueueCandidate(
            expression=to_string(expr),
            fitness=float(fitness),
            queue_score=_admission_queue_score(float(fitness)),
            behavior=behavior,
        )
        for expr, fitness, behavior in pool.elites()
        if fitness >= config.alpha_generator.promotion_min_fitness
    ]
    unresolved = [candidate for candidate in enqueued if candidate.fitness == 0.0]
    if unresolved:
        features = build_feature_list(asset)
        data, prices, _ = _load_generator_data(asset, config, features)
        if data is not None and prices is not None:
            rescored: list[AdmissionQueueCandidate] = []
            for candidate in enqueued:
                fitness = candidate.fitness
                if fitness == 0.0:
                    fitness = _score_expression(
                        candidate.expression,
                        data,
                        prices,
                        config,
                    )
                rescored.append(
                    AdmissionQueueCandidate(
                        expression=candidate.expression,
                        fitness=fitness,
                        queue_score=_admission_queue_score(fitness),
                        behavior=candidate.behavior,
                    )
                )
            enqueued = rescored

    enqueued = [
        candidate
        for candidate in enqueued
        if candidate.fitness >= config.alpha_generator.promotion_min_fitness
    ]
    enqueued.sort(key=lambda candidate: candidate.queue_score, reverse=True)
    enqueued = enqueued[:promote_limit]
    if dry_run or not enqueued:
        return len(enqueued), 0

    store = ManagedAlphaStore(asset_data_dir(asset) / "alpha_registry.db")
    try:
        existing_keys = _existing_enqueue_semantic_keys(store)
        unique_candidates: list[AdmissionQueueCandidate] = []
        for candidate in enqueued:
            semantic_key = expression_semantic_key(candidate.expression)
            if semantic_key in existing_keys:
                continue
            existing_keys.add(semantic_key)
            unique_candidates.append(candidate)
        seeds = [
            CandidateSeed(
                expression=candidate.expression,
                source=f"alpha_generator_{asset.lower()}",
                fitness=candidate.fitness,
                behavior_json={
                    "source": "alpha_generator",
                    "asset": asset,
                    "behavior": [float(x) for x in candidate.behavior.tolist()],
                    "round": None,
                    "enqueue": "manual_discovery_pool",
                },
            )
            for candidate in unique_candidates
        ]
        inserted = store.queue_candidates(seeds)
    finally:
        store.close()
    return len(enqueued), inserted


class AlphaGeneratorDaemon:
    """Pure MAP-Elites alpha generator.

    Each round: generate diverse candidates (random + archive mutation) →
    evaluate once → store in behavior grid → sleep → repeat.
    """

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config
        self.generator_cfg = config.alpha_generator
        self._running = False
        self._round = 0
        self._budget = self.generator_cfg.pop_size

        db_path = asset_data_dir(asset) / "archive.db"
        self.archive = DiscoveryPool.load_from_db(db_path)
        logger.info("Loaded MAP-Elites archive: %d entries", self.archive.size)

    def run(self) -> None:
        self._running = True
        self._setup_signals()
        logger.info(
            "AlphaGeneratorDaemon started: asset=%s, budget=%d, interval=%ds, mutate_ratio=%.1f",
            self.asset, self._budget,
            self.generator_cfg.round_interval, self.generator_cfg.mutate_ratio,
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
        """MAP-Elites round: generate diverse candidates → evaluate → archive."""
        t0 = time.perf_counter()

        all_features = build_feature_list(self.asset)
        k = self.generator_cfg.feature_subset_k
        seed = int(time.time()) ^ self._round

        # Pre-select subset BEFORE loading data
        subset = stratified_feature_subset(all_features, k=k, seed=seed)
        ps = price_signal(self.asset)
        load_features = sorted({ps} | subset)

        data, prices, available_features = self._load_data(load_features)
        if data is None:
            logger.warning("Insufficient data, skipping round")
            return

        n_days = len(prices)
        rng = _random.Random(seed)

        available_subset = frozenset(available_features) - {ps}
        generator = AlphaGenerator(
            available_features,
            feature_subset=available_subset,
            seed=seed,
        )

        engine = BacktestEngine(
            CostModel(self.config.backtest.commission_pct,
                      self.config.backtest.slippage_pct),
            allow_short=self.config.trading.supports_short,
        )

        # Generate candidates: archive mutations + random
        budget = self._budget
        candidates = []
        n_mutated = 0

        if self.archive.size > 0:
            n_mutate = int(budget * self.generator_cfg.mutate_ratio)
            elites = self.archive.sample(n_mutate, rng=rng)
            for entry in elites:
                candidates.append(generator.mutate(entry.expr))
            n_mutated = len(elites)

        n_random = budget - len(candidates)
        candidates.extend(generator.generate_random(
            n_random, max_depth=self.config.generation.max_depth,
        ))

        # Evaluate each candidate once → fitness + behavior → archive
        n_stored = 0
        n_replaced = 0
        queued_candidates: list[AdmissionQueueCandidate] = []

        for expr in candidates:
            try:
                sig = expr.evaluate(data)
                sig = np.asarray(sig, dtype=float)
                if sig.ndim == 0:
                    sig = np.full(n_days, float(sig))
                if len(sig) != n_days:
                    continue
                if not np.all(np.isfinite(sig)):
                    sig = np.where(np.isfinite(sig), sig, 0.0)

                result = engine.run(sig, prices)
                fitness = result.fitness(self.config.fitness_metric)
                if not np.isfinite(fitness) or fitness <= FAILED_FITNESS:
                    continue

                clean_sig = sanitize_signal(sig)
                if clean_sig.ndim == 0:
                    clean_sig = np.full(n_days, float(clean_sig))
                behavior = compute_behavior(clean_sig, expr)

                update = self.archive.store_candidate(
                    expr,
                    behavior,
                    clean_sig,
                    fitness=float(fitness),
                    survival_score=_survival_score(float(fitness), behavior),
                )
                if update.stored:
                    n_stored += 1
                    if update.replaced:
                        n_replaced += 1
                    queued_candidates.append(
                        AdmissionQueueCandidate(
                            expression=to_string(expr),
                            fitness=float(fitness),
                            queue_score=_admission_queue_score(float(fitness)),
                            behavior=behavior,
                        )
                    )
            except Exception:
                continue

        # Persist discovery pool
        db_path = asset_data_dir(self.asset) / "archive.db"
        if n_stored > 0:
            self.archive.save_to_db(db_path)
        n_queued = self._enqueue_admission_queue_candidates(queued_candidates)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Round %d [MAP-Elites]: %d evaluated (%d mutated, %d random), "
            "%d stored (%d replaced), %d queued, "
            "discovery_pool %d/%d (%.1f%%), subset=%d features, %.1fs",
            self._round, len(candidates), n_mutated, len(candidates) - n_mutated,
            n_stored, n_replaced, n_queued,
            self.archive.size, self.archive.capacity,
            self.archive.coverage * 100, len(subset) if subset else 0, elapsed,
        )

    def _enqueue_admission_queue_candidates(
        self,
        candidates: list[AdmissionQueueCandidate],
    ) -> int:
        if not candidates:
            return 0

        limit = self.generator_cfg.promote_per_round
        if limit <= 0:
            return 0

        enqueued = [
            candidate
            for candidate in candidates
            if candidate.fitness >= self.generator_cfg.promotion_min_fitness
        ]
        if not enqueued:
            return 0

        enqueued.sort(key=lambda candidate: candidate.queue_score, reverse=True)
        enqueued = enqueued[:limit]
        store = ManagedAlphaStore(asset_data_dir(self.asset) / "alpha_registry.db")
        try:
            existing_keys = _existing_enqueue_semantic_keys(store)
            unique_candidates: list[AdmissionQueueCandidate] = []
            for candidate in enqueued:
                semantic_key = expression_semantic_key(candidate.expression)
                if semantic_key in existing_keys:
                    continue
                existing_keys.add(semantic_key)
                unique_candidates.append(candidate)
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
                for candidate in unique_candidates
            ]
            return store.queue_candidates(seeds)
        finally:
            store.close()

    def _load_data(
        self, features: list[str],
    ) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None, list[str] | None]:
        return _load_generator_data(self.asset, self.config, features)

    def _check_memory(self) -> None:
        rss_mb = _current_rss_mb()
        limit = self.generator_cfg.memory_limit_mb
        target = self.generator_cfg.pop_size

        if rss_mb > limit:
            old = self._budget
            self._budget = max(_MIN_BUDGET, self._budget // 2)
            if self._budget != old:
                logger.warning(
                    "RSS %.0fMB > limit %dMB, reducing budget %d → %d",
                    rss_mb, limit, old, self._budget,
                )
            return

        recovery_threshold = limit * 0.75
        if rss_mb < recovery_threshold and self._budget < target:
            old = self._budget
            self._budget = min(target, max(old + 1, old * 2))
            logger.info(
                "RSS %.0fMB < recovery threshold %.0fMB, increasing budget %d → %d",
                rss_mb, recovery_threshold, old, self._budget,
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
