"""Intensive random DSL discovery for the current hypotheses-first runtime."""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import time
from datetime import date
from pathlib import Path

import numpy as np

from alpha_os_recovery.config import Config, HYPOTHESES_DB, SIGNAL_CACHE_DB
from alpha_os_recovery.data.eval_universe import (
    load_cached_eval_universe,
    save_eval_universe,
    select_eval_universe,
)
from alpha_os_recovery.data.signal_client import build_signal_client_from_config
from alpha_os_recovery.data.store import DataStore
from alpha_os_recovery.data.universe import (
    CROSS_ASSET_UNIVERSE,
    build_feature_list,
    price_signal,
    stratified_feature_subset,
)
from alpha_os_recovery.dsl import parse, to_string
from alpha_os_recovery.dsl.evaluator import FAILED_FITNESS, sanitize_signal
from alpha_os_recovery.dsl.generator import AlphaGenerator
from alpha_os_recovery.evolution.behavior import compute_behavior
from alpha_os_recovery.evolution.discovery_pool import DiscoveryPool
from alpha_os_recovery.hypotheses.identity import expression_feature_names, expression_semantic_key
from alpha_os_recovery.hypotheses.sleeve_scope import with_scope_asset
from alpha_os_recovery.hypotheses.store import (
    HypothesisKind,
    HypothesisRecord,
    HypothesisStatus,
    HypothesisStore,
)
from alpha_os_recovery.research.cross_asset import evaluate_cross_asset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_DSL_METADATA = {
    "generator": "generate_boost",
    "research_quality_source": "exploratory_unscored",
    "research_quality_status": "unscored",
    "registration_stage": "observation_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an intensive random DSL discovery session against hypotheses.db",
    )
    parser.add_argument("rounds", nargs="?", type=int, default=20)
    parser.add_argument("budget", nargs="?", type=int, default=300)
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/dev/.config/alpha-os/prod.toml",
    )
    return parser.parse_args()


def dsl_hypothesis_id(expression: str) -> str:
    semantic_key = expression_semantic_key(expression)
    digest = hashlib.md5(
        semantic_key.encode(),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return f"dsl_{digest}"


def load_runtime_dsl_records(
    hypothesis_store: HypothesisStore,
    *,
    asset: str,
) -> list[HypothesisRecord]:
    records = hypothesis_store.list_capital_backed(asset=asset)
    return [
        record
        for record in records
        if record.kind == HypothesisKind.DSL and record.expression
    ]


def seed_archive(
    archive: DiscoveryPool,
    *,
    records: list[HypothesisRecord],
    objective: str,
    data: dict[str, np.ndarray],
    prices: np.ndarray,
) -> tuple[int, int]:
    seeded = 0
    skipped = 0
    for record in records:
        try:
            expr = parse(record.expression)
            sig = sanitize_signal(expr.evaluate(data))
            if sig.ndim == 0:
                sig = np.full(len(prices), float(sig))
            behavior = compute_behavior(sig, expr, prices=prices)
            archive.store_candidate(
                expr,
                behavior,
                sig,
                fitness=record.oos_fitness(objective),
            )
            seeded += 1
        except Exception:
            skipped += 1
    return seeded, skipped


def register_candidate(
    hypothesis_store: HypothesisStore,
    *,
    expression: str,
    asset: str,
    fitness: float,
    round_idx: int,
) -> bool:
    hypothesis_id = dsl_hypothesis_id(expression)
    existing = hypothesis_store.get(hypothesis_id)
    if existing is not None:
        return False

    hypothesis_store.register(
        HypothesisRecord(
            hypothesis_id=hypothesis_id,
            kind=HypothesisKind.DSL,
            name=expression[:120],
            definition={"expression": expression},
            status=HypothesisStatus.ACTIVE,
            stake=0.0,
            scope=with_scope_asset(None, asset),
            source="random_dsl",
            metadata={
                **RANDOM_DSL_METADATA,
                "boost_mean_cross_asset_fitness": float(fitness),
                "boost_round": int(round_idx),
                "boost_registered_at": time.time(),
            },
        )
    )
    return True


def fallback_eval_universe(
    data: dict[str, np.ndarray],
    *,
    asset_price_signal: str,
) -> list[str]:
    fallback = [
        signal
        for signal in CROSS_ASSET_UNIVERSE
        if signal in data and int(np.isfinite(data[signal]).sum()) >= 200
    ]
    if fallback:
        return fallback
    if asset_price_signal in data:
        return [asset_price_signal]
    return []


def main() -> None:
    args = parse_args()
    asset = str(args.asset).upper()
    cfg = Config.load(Path(args.config))
    client = build_signal_client_from_config(cfg.api)
    store = DataStore(SIGNAL_CACHE_DB, client)
    hypothesis_store = HypothesisStore(HYPOTHESES_DB)

    seed_records = load_runtime_dsl_records(hypothesis_store, asset=asset)
    seed_required_features = {
        feature
        for record in seed_records
        for feature in expression_feature_names(record.expression)
    }

    all_features = build_feature_list(asset)

    logger.info("Loading data...")
    asset_price_signal = price_signal(asset)
    load_features = sorted(
        {asset_price_signal}
        | set(CROSS_ASSET_UNIVERSE)
        | set(all_features[:100])
        | seed_required_features
    )
    matrix = store.get_matrix(load_features, end=date.today().isoformat())
    store.close()
    if matrix is None or len(matrix) < 200:
        logger.error("Insufficient data")
        return
    data = {col: matrix[col].values for col in matrix.columns}
    prices = data[asset_price_signal]
    first_valid = int(np.argmax(np.isfinite(prices)))
    logger.info(
        "Data loaded: %d rows, %d features, %s from index %d",
        len(matrix),
        len(data),
        asset,
        first_valid,
    )

    eval_universe = load_cached_eval_universe()
    if not eval_universe:
        eval_universe = select_eval_universe(
            data,
            CROSS_ASSET_UNIVERSE,
            n_clusters=20,
            min_finite_days=500,
        )
        if eval_universe:
            save_eval_universe(eval_universe)
    if not eval_universe:
        eval_universe = fallback_eval_universe(
            data,
            asset_price_signal=asset_price_signal,
        )
        if eval_universe:
            logger.warning(
                "Falling back to a non-clustered evaluation universe (%d assets)",
                len(eval_universe),
            )
    if not eval_universe:
        logger.error("No evaluation universe available")
        return
    logger.info("Eval universe (%d): %s", len(eval_universe), eval_universe)

    archive = DiscoveryPool()
    try:
        seeded, skipped_seed = seed_archive(
            archive,
            records=seed_records,
            objective=cfg.portfolio.objective,
            data=data,
            prices=prices,
        )
        logger.info(
            "Seeded archive with %d runtime hypotheses (skipped=%d, archive size=%d)",
            seeded,
            skipped_seed,
            archive.size,
        )

        total_candidates = 0
        total_positive = 0
        total_registered = 0

        for round_idx in range(args.rounds):
            started = time.perf_counter()
            seed = int(time.time()) ^ round_idx
            rng = random.Random(seed)

            subset = stratified_feature_subset(
                all_features,
                k=cfg.alpha_generator.feature_subset_k,
                seed=seed,
            )
            available = [feature for feature in subset if feature in data]

            generator = AlphaGenerator(
                list(data.keys()),
                feature_subset=frozenset(available),
                seed=seed,
            )

            candidates = []
            if archive.size > 0:
                n_mutate = int(args.budget * 0.7)
                elites = archive.sample(min(n_mutate, archive.size), rng=rng)
                for entry in elites:
                    candidates.append(generator.mutate(entry.expr))

            n_random = args.budget - len(candidates)
            candidates.extend(
                generator.generate_random(
                    n_random,
                    max_depth=cfg.generation.max_depth,
                )
            )

            positive = 0
            registered = 0

            for expr in candidates:
                try:
                    expr_str = to_string(expr)
                    per_asset = evaluate_cross_asset(
                        expr_str,
                        data,
                        eval_universe,
                        fitness_metric=cfg.portfolio.objective,
                        commission_pct=cfg.backtest.commission_pct,
                        slippage_pct=cfg.backtest.slippage_pct,
                        allow_short=True,
                        benchmark_assets=cfg.backtest.benchmark_assets,
                    )
                    if not per_asset:
                        continue
                    fitness = float(np.mean(list(per_asset.values())))
                    if not np.isfinite(fitness) or fitness <= FAILED_FITNESS:
                        continue

                    positive += 1

                    sig = sanitize_signal(expr.evaluate(data))
                    if sig.ndim == 0:
                        sig = np.full(len(prices), float(sig))
                    behavior = compute_behavior(sig, expr, prices=prices)
                    archive.store_candidate(expr, behavior, sig, fitness=fitness)

                    if fitness > 0 and register_candidate(
                        hypothesis_store,
                        expression=expr_str,
                        asset=asset,
                        fitness=fitness,
                        round_idx=round_idx + 1,
                    ):
                        registered += 1
                except Exception:
                    continue

            total_candidates += len(candidates)
            total_positive += positive
            total_registered += registered
            elapsed = time.perf_counter() - started

            logger.info(
                "Round %d/%d: %d candidates, %d positive (%.1f%%), %d registered, %.1fs, archive=%d",
                round_idx + 1,
                args.rounds,
                len(candidates),
                positive,
                100 * positive / max(len(candidates), 1),
                registered,
                elapsed,
                archive.size,
            )

        logger.info(
            "DONE: %d rounds, %d total candidates, %d positive (%.1f%%), %d registered to hypotheses.db",
            args.rounds,
            total_candidates,
            total_positive,
            100 * total_positive / max(total_candidates, 1),
            total_registered,
        )
    finally:
        hypothesis_store.close()


if __name__ == "__main__":
    main()
