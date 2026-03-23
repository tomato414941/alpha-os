"""Discovery pool enqueue utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..dsl.evaluator import FAILED_FITNESS, sanitize_signal
from ..hypotheses.identity import expression_semantic_key
from ..backtest.cost_model import CostModel
from ..backtest.engine import BacktestEngine
from ..config import Config, SIGNAL_CACHE_DB, asset_data_dir
from ..data.signal_client import build_signal_client_from_config
from ..dsl import parse, to_string
from ..data.universe import build_feature_list, price_signal
from ..evolution.discovery_pool import DiscoveryPool
from ..legacy.managed_alphas import CandidateSeed, ManagedAlphaStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdmissionQueueCandidate:
    expression: str
    fitness: float
    queue_score: float
    behavior: np.ndarray


def _admission_queue_score(fitness: float) -> float:
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


def _load_generator_data(
    asset: str,
    config: Config,
    features: list[str],
) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None, list[str] | None]:
    from ..data.store import DataStore

    db_path = SIGNAL_CACHE_DB
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
    matrix = matrix.fillna(0)

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
    benchmark_returns: np.ndarray | None = None,
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
        result = engine.run(sig, prices, benchmark_returns=benchmark_returns)
        value = result.fitness(config.portfolio.objective)
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
    pool = DiscoveryPool.load_from_db(asset_data_dir(asset) / "discovery_pool.db")
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
            bm_returns = None
            if config.backtest.benchmark_assets:
                from alpha_os.backtest.benchmark import build_benchmark_returns
                bm_returns = build_benchmark_returns(data, config.backtest.benchmark_assets)
                if len(bm_returns) == 0:
                    bm_returns = None
            rescored: list[AdmissionQueueCandidate] = []
            for candidate in enqueued:
                fitness = candidate.fitness
                if fitness == 0.0:
                    fitness = _score_expression(
                        candidate.expression,
                        data,
                        prices,
                        config,
                        benchmark_returns=bm_returns,
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
