"""Intensive alpha generation session — run many rounds with high budget."""
import sys
import time
import numpy as np
from pathlib import Path

from alpha_os.config import Config, SIGNAL_CACHE_DB, asset_data_dir
from alpha_os.data.store import DataStore
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.universe import build_feature_list, price_signal, stratified_feature_subset
from alpha_os.dsl import parse, to_string
from alpha_os.dsl.generator import AlphaGenerator
from alpha_os.dsl.evaluator import FAILED_FITNESS, sanitize_signal
from alpha_os.evolution.behavior import compute_behavior
from alpha_os.evolution.discovery_pool import DiscoveryPool
from alpha_os.legacy.admission_queue import CandidateSeed
from alpha_os.legacy.managed_alphas import ManagedAlphaStore
from alpha_os.research.cross_asset import evaluate_cross_asset
from datetime import date
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_UNIVERSE = None  # auto-selected at runtime

N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 300


def main():
    cfg = Config.load(Path("/home/dev/.config/alpha-os/prod.toml"))
    client = build_signal_client_from_config(cfg.api)
    store = DataStore(SIGNAL_CACHE_DB, client)

    all_features = build_feature_list("BTC")

    # Load data once
    logger.info("Loading data...")
    from alpha_os.data.universe import CROSS_ASSET_UNIVERSE
    ps = price_signal("BTC")
    load_features = sorted({ps} | set(CROSS_ASSET_UNIVERSE) | set(all_features[:100]))
    matrix = store.get_matrix(load_features, end=date.today().isoformat())
    if matrix is None or len(matrix) < 200:
        logger.error("Insufficient data")
        return
    data = {col: matrix[col].values for col in matrix.columns}
    prices = data[ps]
    first_valid = np.argmax(np.isfinite(prices))
    logger.info("Data loaded: %d rows, %d features, BTC from index %d", len(matrix), len(data), first_valid)

    # Use cached eval universe; recompute only if missing
    global EVAL_UNIVERSE
    from alpha_os.data.eval_universe import load_cached_eval_universe, save_eval_universe, select_eval_universe
    EVAL_UNIVERSE = load_cached_eval_universe()
    if not EVAL_UNIVERSE:
        EVAL_UNIVERSE = select_eval_universe(data, CROSS_ASSET_UNIVERSE, n_clusters=20, min_finite_days=500)
        if EVAL_UNIVERSE:
            save_eval_universe(EVAL_UNIVERSE)
    logger.info("Eval universe (%d): %s", len(EVAL_UNIVERSE), EVAL_UNIVERSE)

    archive = DiscoveryPool()
    adir = asset_data_dir("BTC")
    reg = ManagedAlphaStore(db_path=adir / "alpha_registry.db")

    # Seed archive with existing deployed alphas (known positive residual fitness)
    deployed_ids = reg.deployed_alpha_ids()
    n_seeded = 0
    for aid in deployed_ids:
        record = reg.get(aid)
        if not record:
            continue
        try:
            expr = parse(record.expression)
            sig = sanitize_signal(expr.evaluate(data))
            if sig.ndim == 0:
                sig = np.full(len(prices), float(sig))
            behavior = compute_behavior(sig, expr, prices=prices)
            archive.store_candidate(expr, behavior, sig, fitness=record.fitness)
            n_seeded += 1
        except Exception:
            continue
    logger.info("Seeded archive with %d deployed alphas (archive size: %d)", n_seeded, archive.size)

    total_candidates = 0
    total_positive = 0
    total_queued = 0

    for round_idx in range(N_ROUNDS):
        t0 = time.perf_counter()
        seed = int(time.time()) ^ round_idx
        rng = random.Random(seed)

        k = cfg.alpha_generator.feature_subset_k
        subset = stratified_feature_subset(all_features, k=k, seed=seed)
        available = [f for f in list(subset) if f in data]

        generator = AlphaGenerator(
            list(data.keys()),
            feature_subset=frozenset(available),
            seed=seed,
        )

        # Generate candidates: 70% archive mutations + 30% random
        candidates = []
        if archive.size > 0:
            n_mutate = int(BUDGET * 0.7)
            elites = archive.sample(min(n_mutate, archive.size), rng=rng)
            for entry in elites:
                candidates.append(generator.mutate(entry.expr))

        n_random = BUDGET - len(candidates)
        candidates.extend(generator.generate_random(n_random, max_depth=cfg.generation.max_depth))

        n_positive = 0
        queued = []

        for expr in candidates:
            try:
                expr_str = to_string(expr)
                per_asset = evaluate_cross_asset(
                    expr_str, data, EVAL_UNIVERSE,
                    fitness_metric=cfg.fitness_metric,
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

                n_positive += 1

                sig = sanitize_signal(expr.evaluate(data))
                if sig.ndim == 0:
                    sig = np.full(len(prices), float(sig))
                behavior = compute_behavior(sig, expr, prices=prices)
                archive.store_candidate(expr, behavior, sig, fitness=fitness)

                if fitness > 0:
                    queued.append(CandidateSeed(
                        expression=expr_str,
                        source="generate_boost",
                        fitness=fitness,
                    ))
            except Exception:
                continue

        if queued:
            n_q = reg.queue_candidates(queued)
            total_queued += n_q

        total_candidates += len(candidates)
        total_positive += n_positive
        elapsed = time.perf_counter() - t0

        logger.info(
            "Round %d/%d: %d candidates, %d positive (%.1f%%), %d queued, %.1fs, archive=%d",
            round_idx + 1, N_ROUNDS, len(candidates), n_positive,
            100 * n_positive / max(len(candidates), 1),
            len(queued), elapsed, archive.size,
        )

    reg.close()

    logger.info(
        "DONE: %d rounds, %d total candidates, %d positive (%.1f%%), %d queued to registry",
        N_ROUNDS, total_candidates, total_positive,
        100 * total_positive / max(total_candidates, 1),
        total_queued,
    )


if __name__ == "__main__":
    main()
