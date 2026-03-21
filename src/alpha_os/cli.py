"""CLI for alpha-os: generate, backtest, validate alpha factors."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.config import Config, DATA_DIR, asset_data_dir
from alpha_os.runtime_lock import RuntimeLockBusy, hold_runtime_lock, runtime_lock_path
from alpha_os.alpha.evaluator import FAILED_FITNESS, sanitize_signal
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.universe import is_crypto, is_equity, infer_venue, price_signal, build_feature_list, build_hourly_feature_list
from alpha_os.dsl import parse, to_string
from alpha_os.runtime_profile import build_runtime_profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-os",
        description="Agentic Alpha OS — generate, backtest, validate alpha factors",
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", help="Generate alpha expressions")
    gen.add_argument("--count", type=int, default=5000)
    gen.add_argument("--asset", type=str, default="NVDA")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--config", type=str, default=None)

    # backtest
    bt = sub.add_parser("backtest", help="Backtest generated alphas")
    bt.add_argument("--count", type=int, default=5000)
    bt.add_argument("--top", type=int, default=20)
    bt.add_argument("--asset", type=str, default="NVDA")
    bt.add_argument("--days", type=int, default=500,
                    help="Number of days (--synthetic only)")
    bt.add_argument("--seed", type=int, default=42)
    bt.add_argument("--synthetic", action="store_true",
                    help="Use synthetic random-walk data instead of real data")
    bt.add_argument("--eval-window", type=int, default=0,
                    help="Evaluation window in days (0=all data, e.g. 200 for recent)")
    bt.add_argument("--config", type=str, default=None)

    # evolve
    evo = sub.add_parser("evolve", help="Evolve alphas via GP + MAP-Elites")
    evo.add_argument("--pop-size", type=int, default=200)
    evo.add_argument("--generations", type=int, default=30)
    evo.add_argument("--asset", type=str, default="NVDA")
    evo.add_argument("--days", type=int, default=500,
                    help="Number of days (--synthetic only)")
    evo.add_argument("--top", type=int, default=20)
    evo.add_argument("--seed", type=int, default=42)
    evo.add_argument("--synthetic", action="store_true",
                    help="Use synthetic random-walk data instead of real data")
    evo.add_argument("--eval-window", type=int, default=0,
                    help="Evaluation window in days (0=all data, e.g. 200 for recent)")
    evo.add_argument("--layer", type=int, default=3, choices=[2, 3],
                    help="Alpha layer: 2=hourly tactical, 3=daily strategic (default)")
    evo.add_argument("--config", type=str, default=None)

    # validate
    val = sub.add_parser("validate", help="Validate an alpha with purged WF CV")
    val.add_argument("--expr", type=str, required=True)
    val.add_argument("--asset", type=str, default="NVDA")
    val.add_argument("--days", type=int, default=500,
                    help="Number of days (--synthetic only)")
    val.add_argument("--seed", type=int, default=42)
    val.add_argument("--synthetic", action="store_true",
                    help="Use synthetic random-walk data instead of real data")
    val.add_argument("--eval-window", type=int, default=0,
                    help="Evaluation window in days (0=all data, e.g. 200 for recent)")
    val.add_argument("--layer", type=int, default=3, choices=[2, 3],
                    help="Alpha layer: 2=hourly tactical, 3=daily strategic (default)")
    val.add_argument("--config", type=str, default=None)

    # evaluate — multi-horizon IC evaluation
    eva = sub.add_parser("evaluate", help="Evaluate expression with multi-horizon IC")
    eva.add_argument("--expr", type=str, required=True, help="DSL expression string")
    eva.add_argument("--config", type=str, default=None)

    # submit — submit expression to admission queue
    smt = sub.add_parser("submit", help="Submit expression to admission queue")
    smt.add_argument("--expr", type=str, required=True, help="DSL expression string")
    smt.add_argument("--asset", type=str, default="BTC")
    smt.add_argument("--config", type=str, default=None)

    # produce-predictions
    pp = sub.add_parser("produce-predictions", help="Evaluate active alphas and write to prediction store")
    pp.add_argument("--asset", type=str, default="BTC")
    pp.add_argument("--config", type=str, default=None)

    # produce-classical
    pc = sub.add_parser("produce-classical", help="Compute classical indicators and write to prediction store")
    pc.add_argument("--config", type=str, default=None)

    # paper
    ppr = sub.add_parser("paper", help="Paper trade with adopted alphas")
    ppr.add_argument("--once", action="store_true", help="Run one cycle and exit")
    ppr.add_argument("--schedule", action="store_true", help="Run on interval")
    ppr.add_argument("--summary", action="store_true", help="Print summary and exit")
    ppr.add_argument("--interval", type=int, default=None,
                     help="Override check_interval in seconds (default: from config)")
    ppr.add_argument("--replay", action="store_true",
                     help="Run historical replay over date range")
    ppr.add_argument("--start", type=str, default=None,
                     help="Start date for replay (ISO format, e.g. 2025-06-01)")
    ppr.add_argument("--end", type=str, default=None,
                     help="End date for replay (ISO format, e.g. 2026-02-25)")
    ppr.add_argument("--sizing-mode", type=str, default="runtime",
                     choices=["runtime", "raw_mean", "compare"],
                     help="Replay sizing mode: runtime logic, raw_mean baseline, or compare both")
    ppr.add_argument("--tactical", action="store_true",
                     help="Enable Layer 2 tactical modulation")
    ppr.add_argument("--asset", type=str, default="BTC")
    ppr.add_argument("--config", type=str, default=None)

    # trade
    trd = sub.add_parser("trade", help="Trade on Binance (testnet by default)")
    trd.add_argument("--once", action="store_true", help="Run one cycle and exit")
    trd.add_argument("--schedule", action="store_true", help="Run on interval")
    trd.add_argument("--summary", action="store_true", help="Print summary and exit")
    trd.add_argument("--interval", type=int, default=None,
                     help="Override check_interval in seconds (default: from config)")
    trd.add_argument("--real", action="store_true",
                     help="Use real Binance (default is testnet)")
    trd.add_argument("--capital", type=float, default=10000.0,
                     help="Initial capital for tracking (default: 10000)")
    trd.add_argument("--asset", type=str, default="BTC")
    trd.add_argument("--assets", type=str, default=None,
                     help="Comma-separated asset list (e.g. BTC,ETH,SOL)")
    trd.add_argument("--config", type=str, default=None)
    trd.add_argument("--evolve-interval", type=int, default=86400,
                     help="Alpha evolution interval in seconds (default: 86400=24h, 0=disable)")
    trd.add_argument("--pop-size", type=int, default=200,
                     help="GP population size for evolution")
    trd.add_argument("--generations", type=int, default=30,
                     help="GP generations per evolution cycle")
    trd.add_argument("--event-driven", action="store_true",
                     help="Use event-driven execution instead of fixed interval")
    trd.add_argument("--debounce", type=int, default=None,
                     help="Min seconds between event-triggered evaluations (default: from config)")
    trd.add_argument("--venue", type=str, default=None,
                     choices=["binance", "alpaca", "polymarket", "paper"],
                     help="Trading venue (default: auto-detect from asset)")

    # cross-trade (cross-sectional trading)
    xst = sub.add_parser("cross-trade", help="Cross-sectional multi-asset trading")
    xst.add_argument("--assets", type=str, required=True,
                     help="Comma-separated tradeable assets (e.g. BTC,ETH,SOL)")
    xst.add_argument("--once", action="store_true", help="Run one cycle and exit")
    xst.add_argument("--schedule", action="store_true", help="Run on interval")
    xst.add_argument("--interval", type=int, default=None)
    xst.add_argument("--capital", type=float, default=10000.0)
    xst.add_argument("--real", action="store_true")
    xst.add_argument("--config", type=str, default=None)

    ugen = sub.add_parser(
        "unified-generator",
        help="Run cross-asset alpha generation daemon (evaluates across all assets)",
    )
    ugen.add_argument("--config", type=str, default=None)

    pg = sub.add_parser(
        "enqueue-discovery-pool",
        help="Enqueue top discovery-pool entries into the admission queue",
    )
    pg.add_argument("--asset", type=str, default="BTC")
    pg.add_argument("--config", type=str, default=None)
    pg.add_argument("--limit", type=int, default=None)
    pg.add_argument("--dry-run", action="store_true")

    # admission-daemon (Pipeline v2)
    adm_d = sub.add_parser("admission-daemon", help="Run candidate admission daemon")
    adm_d.add_argument("--asset", type=str, default="BTC")
    adm_d.add_argument("--config", type=str, default=None)

    psc = sub.add_parser(
        "prune-stale-candidates",
        help="Reject stale pending candidates outside the active discovery/manual sources",
    )
    psc.add_argument("--asset", type=str, default="BTC")
    psc.add_argument("--max-age-days", type=int, default=7)
    psc.add_argument("--dry-run", action="store_true")

    # lifecycle (Pipeline v2)
    lc_d = sub.add_parser("lifecycle", help="Run daily lifecycle evaluation (oneshot)")
    lc_d.add_argument("--asset", type=str, default="BTC")
    lc_d.add_argument("--config", type=str, default=None)

    # rebuild-managed-alphas
    rb = sub.add_parser(
        "rebuild-managed-alphas",
        help="Replay admission gates and rebuild managed alpha states",
    )
    rb.add_argument("--asset", type=str, default="BTC")
    rb.add_argument("--config", type=str, default=None)
    rb.add_argument(
        "--source",
        choices=["candidates", "alphas"],
        default="candidates",
        help="Source records used to rebuild managed alpha states",
    )
    rb.add_argument(
        "--fail-state",
        choices=["rejected", "dormant"],
        default="rejected",
        help="State assigned to records that fail the admission gate",
    )
    rb.add_argument("--dry-run", action="store_true", help="Print counts without writing")
    rb.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip copying alpha_registry.db before rewrite",
    )

    # refresh-deployed-alphas
    rda = sub.add_parser(
        "refresh-deployed-alphas",
        help="Refresh deployed alphas from managed alphas",
    )
    rda.add_argument("--asset", type=str, default="BTC")
    rda.add_argument("--config", type=str, default=None)
    rda.add_argument("--dry-run", action="store_true", help="Print the plan without writing")
    rda.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip copying alpha_registry.db before rewrite",
    )

    # prune-managed-alpha-duplicates
    prd = sub.add_parser(
        "prune-managed-alpha-duplicates",
        help="Demote duplicate active managed alphas and refresh deployed alphas",
    )
    prd.add_argument("--asset", type=str, default="BTC")
    prd.add_argument("--config", type=str, default=None)
    prd.add_argument("--dry-run", action="store_true", help="Print the plan without writing")
    prd.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip copying alpha_registry.db before rewrite",
    )
    prd.add_argument(
        "--no-refresh-deployed",
        action="store_true",
        help="Do not refresh deployed alphas after demoting duplicates",
    )

    # replay-experiment
    rex = sub.add_parser(
        "replay-experiment",
        help="Run a named replay experiment and persist the artifact",
    )
    rex.add_argument("--name", required=True, help="Experiment name")
    rex.add_argument("--asset", type=str, default="BTC")
    rex.add_argument("--config", type=str, default=None)
    rex.add_argument("--start", required=True, help="Replay start date (YYYY-MM-DD)")
    rex.add_argument("--end", required=True, help="Replay end date (YYYY-MM-DD)")
    rex.add_argument(
        "--managed-alpha-mode",
        choices=["current", "admission"],
        default="current",
        help="Use the current managed-alpha set as-is or rebuild it from admission rules first",
    )
    rex.add_argument(
        "--source",
        choices=["alphas", "candidates"],
        default="candidates",
        help="Admission replay source when --managed-alpha-mode=admission",
    )
    rex.add_argument(
        "--fail-state",
        choices=["rejected", "dormant"],
        default="rejected",
        help="Fallback state for records that fail admission replay",
    )
    rex.add_argument(
        "--deployment-mode",
        choices=["current", "refresh"],
        default="current",
        help="Use the current deployed alpha set or refresh it inside the experiment",
    )
    rex.add_argument(
        "--sizing-mode",
        type=str,
        default="runtime",
        choices=["runtime", "raw_mean"],
        help="Replay sizing mode",
    )
    rex.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override merged config via dotted path, e.g. lifecycle.candidate_quality_min=1.10",
    )
    rex.add_argument("--notes", default="", help="Optional experiment notes")

    # replay-matrix
    rmx = sub.add_parser(
        "replay-matrix",
        help="Run a TOML-defined replay experiment matrix",
    )
    rmx.add_argument("--manifest", required=True, help="Path to TOML matrix manifest")
    rmx.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers for historical replay runs",
    )

    # testnet-readiness
    tnr = sub.add_parser("testnet-readiness", help="Check Phase 4 testnet readiness status")
    tnr.add_argument("--reports", action="store_true", help="Show all daily reports")
    tnr.add_argument("--slippage", action="store_true", help="Show slippage distribution")
    tnr.add_argument("--latency", action="store_true", help="Show fill latency distribution")
    tnr.add_argument("--reset", action="store_true", help="Reset consecutive day counter")
    tnr.add_argument("--asset", type=str, default="BTC")
    tnr.add_argument("--config", type=str, default=None)

    # runtime-status
    rst = sub.add_parser(
        "runtime-status",
        help="Show current runtime observation status from registry and readiness files",
    )
    rst.add_argument("--asset", type=str, default="BTC")
    rst.add_argument("--config", type=str, default=None)

    afl = sub.add_parser(
        "alpha-funnel",
        help="Show discovery-pool to deployed-alpha funnel counts",
    )
    afl.add_argument("--asset", type=str, default="BTC")

    # seed-handcrafted
    shc = sub.add_parser(
        "seed-handcrafted",
        help="Queue hand-crafted alpha candidates into the admission queue",
    )
    shc.add_argument("--asset", type=str, default="BTC")
    shc.add_argument(
        "--alpha-set",
        type=str,
        default="baseline",
        help="Named handcrafted alpha set for the asset",
    )
    shc.add_argument("--list", action="store_true", help="List available sets and exit")
    shc.add_argument("--dry-run", action="store_true", help="Print expressions without writing")

    # analyze-diversity
    adv = sub.add_parser(
        "analyze-diversity",
        help="Analyze alpha diversity across signal, structure, and feature families",
    )
    adv.add_argument("--asset", type=str, default="BTC")
    adv.add_argument("--config", type=str, default=None)
    adv.add_argument(
        "--scope",
        choices=["deployed", "active"],
        default="deployed",
        help="Analyze the deployed alpha set or the registry-active pool",
    )
    adv.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional record cap (0 = full scope)",
    )
    adv.add_argument(
        "--metric",
        choices=["sharpe", "log_growth"],
        default="sharpe",
        help="Ordering metric when limiting the registry-active pool",
    )
    adv.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Signal lookback window used for correlation analysis",
    )
    adv.add_argument(
        "--top-pairs",
        type=int,
        default=10,
        help="Number of most redundant pairs to print",
    )
    adv.add_argument("--json", action="store_true", help="Print full report as JSON")

    return parser


def _load_config(config_path: str | None) -> Config:
    cfg = Config.load(Path(config_path)) if config_path else Config.load()
    os.environ["ALPHA_OS_SIGNAL_NOISE_URL"] = cfg.api.base_url
    return cfg


def _load_runtime_observation_config(config_path: str | None) -> Config:
    if config_path:
        cfg = Config.load(Path(config_path))
        os.environ["ALPHA_OS_SIGNAL_NOISE_URL"] = cfg.api.base_url
        return cfg
    user_prod = Path.home() / ".config" / "alpha-os" / "prod.toml"
    if user_prod.exists():
        cfg = Config.load(user_prod)
        os.environ["ALPHA_OS_SIGNAL_NOISE_URL"] = cfg.api.base_url
        return cfg
    cfg = Config.load()
    os.environ["ALPHA_OS_SIGNAL_NOISE_URL"] = cfg.api.base_url
    return cfg


def _normalize_trade_config(cfg: Config) -> list[str]:
    """Trade runtime now respects the configured profile as-is."""
    return []


def _make_features(asset: str) -> list[str]:
    """Feature names available for alpha generation."""
    return build_feature_list(asset)


def _synthetic_data(features: list[str], n_days: int, seed: int) -> dict[str, np.ndarray]:
    """Generate synthetic price/signal data for offline testing."""
    print("[SYNTHETIC] Using synthetic random-walk data — not suitable for real decisions")
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for feat in features:
        drift = rng.uniform(-0.0005, 0.001)
        vol = rng.uniform(0.005, 0.03)
        returns = rng.normal(drift, vol, n_days)
        prices = 100.0 * np.cumprod(1.0 + returns)
        data[feat] = prices
    return data


def _real_data(
    features: list[str], config: Config, eval_window: int = 0,
    resolution: str = "1d",
) -> tuple[dict[str, np.ndarray], int]:
    """Load real data: cache-first, import from signal-noise, sync from API."""
    from alpha_os.data.store import DataStore

    db_name = "alpha_cache_l2.db" if resolution != "1d" else "alpha_cache.db"
    db_path = DATA_DIR / db_name

    client = build_signal_client_from_config(config.api)
    store = DataStore(db_path, client)

    # Sync from REST API (incremental, best-effort)
    try:
        if client.health():
            print(f"Syncing from {config.api.base_url} (resolution={resolution}) ...")
            store.sync(features, resolution=resolution)
        else:
            print(f"API unavailable at {config.api.base_url} — using cache")
    except Exception as exc:
        print(f"API sync failed — using cache: {exc}")

    matrix = store.get_matrix(features, resolution=resolution)

    # Only require price signal (first feature) to be present
    price_col = features[0]
    if price_col in matrix.columns:
        matrix = matrix[matrix[price_col].notna()]

    matrix = matrix.fillna(0)

    available = [f for f in features if f in matrix.columns and not (matrix[f] == 0).all()]
    missing = [f for f in features if f not in available]
    if missing:
        print(f"Missing/empty: {len(missing)} signals")
    print(f"Available: {len(available)} signals")

    store.close()

    if len(matrix) < 60:
        raise RuntimeError(
            f"Insufficient data: {len(matrix)} rows (need >= 60). "
            f"Missing signals: {missing}. "
            "Run with API access first to populate cache, "
            "or use --synthetic for testing."
        )

    data = {col: matrix[col].values for col in matrix.columns}
    n_days = len(matrix)
    print(f"Loaded {n_days} daily data points, {len(features)} features")

    # Apply evaluation window
    if eval_window > 0 and n_days > eval_window:
        data = {k: v[-eval_window:] for k, v in data.items()}
        n_days = eval_window
        print(f"Eval window: using last {eval_window} days")

    return data, n_days


def cmd_generate(args: argparse.Namespace) -> None:
    features = _make_features(args.asset)
    from alpha_os.dsl.generator import AlphaGenerator
    gen = AlphaGenerator(features=features, seed=args.seed)

    t0 = time.perf_counter()
    alphas = gen.generate_random(args.count, max_depth=3)
    templates = gen.generate_from_templates()
    elapsed = time.perf_counter() - t0

    total = len(alphas) + len(templates)
    print(f"Generated {total} alphas ({len(alphas)} random + {len(templates)} template) in {elapsed:.2f}s")
    print("\nSample random alphas:")
    for a in alphas[:5]:
        print(f"  {to_string(a)}")
    print("\nSample template alphas:")
    for a in templates[:5]:
        print(f"  {to_string(a)}")


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    features = _make_features(args.asset)
    from alpha_os.dsl.generator import AlphaGenerator
    gen = AlphaGenerator(features=features, seed=args.seed)

    # Generate alphas
    t0 = time.perf_counter()
    alphas = gen.generate_random(args.count, max_depth=3)
    templates = gen.generate_from_templates()
    all_alphas = alphas + templates
    gen_time = time.perf_counter() - t0

    # Data source
    if args.synthetic:
        data = _synthetic_data(features, args.days, seed=args.seed + 1000)
        n_days = args.days
    else:
        data, n_days = _real_data(features, cfg, eval_window=args.eval_window)

    price_feat = features[0]
    prices = data[price_feat]

    # Evaluate all alphas to signal arrays
    t1 = time.perf_counter()
    signals = []
    valid_alphas = []
    for expr in all_alphas:
        try:
            sig = expr.evaluate(data)
            if isinstance(sig, (int, float, np.floating)):
                sig = np.full(n_days, float(sig))
            if len(sig) != n_days:
                continue
            if np.all(np.isnan(sig)):
                continue
            if not np.all(np.isfinite(sig)):
                sig = np.where(np.isfinite(sig), sig, 0.0)
            signals.append(sig)
            valid_alphas.append(expr)
        except Exception:
            continue
    eval_time = time.perf_counter() - t1

    if not signals:
        print("No valid alphas to backtest.")
        return

    # Batch backtest
    t2 = time.perf_counter()
    sig_matrix = np.array(signals)
    engine = BacktestEngine(
        CostModel(cfg.backtest.commission_pct, cfg.backtest.slippage_pct),
        allow_short=cfg.trading.supports_short,
    )
    results = engine.run_batch(
        sig_matrix, prices,
        alpha_ids=[f"alpha_{i}" for i in range(len(signals))],
    )
    bt_time = time.perf_counter() - t2

    # Rank by Sharpe (NaN → bottom)
    ranked = sorted(
        zip(valid_alphas, results),
        key=lambda x: x[1].sharpe if np.isfinite(x[1].sharpe) else FAILED_FITNESS,
        reverse=True,
    )

    total_time = time.perf_counter() - t0
    print(f"Pipeline: {len(all_alphas)} generated, {len(signals)} valid, backtested in {total_time:.2f}s")
    print(f"  Generate: {gen_time:.2f}s | Evaluate: {eval_time:.2f}s | Backtest: {bt_time:.2f}s")
    print(f"\nTop {args.top} alphas by Sharpe ratio:")
    print(
        f"{'Rank':>4}  {'Sharpe':>8}  {'Return':>8}  {'MaxDD':>8}  {'CVaR95':>8}  "
        f"{'TailHit':>8}  {'Turnover':>8}  Expression"
    )
    print("-" * 120)
    for i, (expr, res) in enumerate(ranked[: args.top]):
        print(
            f"{i + 1:>4}  {res.sharpe:>8.3f}  {res.annual_return:>7.1%}  "
            f"{res.max_drawdown:>7.1%}  {res.cvar_95:>8.3%}  {res.tail_hit_rate:>7.1%}  "
            f"{res.turnover:>8.3f}  {to_string(expr)}"
        )


def cmd_evolve(args: argparse.Namespace) -> None:
    from alpha_os.evolution.discovery_pool import DiscoveryPool
    from alpha_os.evolution.behavior import compute_behavior
    from alpha_os.evolution.gp import GPConfig, GPEvolver

    cfg = _load_config(args.config)
    layer = getattr(args, "layer", 3)
    if layer == 2:
        features = build_hourly_feature_list(args.asset)
    else:
        features = _make_features(args.asset)

    if args.synthetic:
        n_days = args.days
        data = _synthetic_data(features, n_days, seed=args.seed + 1000)
    else:
        resolution = "1h" if layer == 2 else "1d"
        data, n_days = _real_data(features, cfg, eval_window=args.eval_window, resolution=resolution)

    price_feat = features[0]
    prices = data[price_feat]

    engine = BacktestEngine(
        CostModel(cfg.backtest.commission_pct, cfg.backtest.slippage_pct),
        allow_short=cfg.trading.supports_short,
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
            return result.sharpe if np.isfinite(result.sharpe) else FAILED_FITNESS
        except Exception:
            return FAILED_FITNESS

    gp_cfg = GPConfig(
        pop_size=args.pop_size,
        n_generations=args.generations,
        max_depth=cfg.generation.max_depth,
        bloat_penalty=cfg.generation.bloat_penalty,
        depth_penalty=cfg.generation.depth_penalty,
        similarity_penalty=cfg.generation.similarity_penalty,
    )
    evolver = GPEvolver(features, evaluate_fn, config=gp_cfg, seed=args.seed)

    t0 = time.perf_counter()
    results = evolver.run()
    evolve_time = time.perf_counter() - t0

    # Fill discovery pool
    pool = DiscoveryPool()
    live_signals: list[np.ndarray] = []
    added = 0
    for expr, fitness in results:
        try:
            sig = expr.evaluate(data)
            sig = sanitize_signal(sig)
            if sig.ndim == 0:
                sig = np.full(n_days, float(sig))
            behavior = compute_behavior(sig, expr, prices=prices)
            if pool.add(expr, fitness, behavior):
                added += 1
                live_signals.append(sig)
        except Exception:
            continue

    total_time = time.perf_counter() - t0
    layer_label = f"Layer {layer}" if layer == 2 else "Layer 3"
    print(f"Evolution ({layer_label}): {len(results)} unique alphas in {evolve_time:.2f}s")
    print(f"Pool: {pool.size}/{pool.capacity} cells filled ({pool.coverage:.1%} coverage)")
    print(f"Total time: {total_time:.2f}s\n")

    top = pool.best(args.top)
    print(f"Top {min(args.top, len(top))} alphas by fitness:")
    print(f"{'Rank':>4}  {'Fitness':>8}  Expression")
    print("-" * 70)
    for i, (expr, fit) in enumerate(top):
        print(f"{i + 1:>4}  {fit:>8.3f}  {to_string(expr)}")


def cmd_validate(args: argparse.Namespace) -> None:
    from alpha_os.validation.purged_cv import purged_walk_forward

    cfg = _load_config(args.config)
    layer = getattr(args, "layer", 3)

    if layer == 2:
        features = build_hourly_feature_list(args.asset)
    else:
        features = _make_features(args.asset)

    # Parse expression
    expr = parse(args.expr)
    print(f"Alpha: {to_string(expr)}")

    if args.synthetic:
        n_days = args.days
        data = _synthetic_data(features, n_days, seed=args.seed + 1000)
    else:
        resolution = "1h" if layer == 2 else "1d"
        data, n_days = _real_data(features, cfg, eval_window=args.eval_window,
                                   resolution=resolution)

    price_feat = features[0]
    prices = data[price_feat]

    # Evaluate
    sig = expr.evaluate(data)
    if isinstance(sig, (int, float, np.floating)):
        sig = np.full(n_days, float(sig))

    # Purged Walk-Forward CV
    engine = BacktestEngine(
        CostModel(cfg.backtest.commission_pct, cfg.backtest.slippage_pct),
        allow_short=cfg.trading.supports_short,
    )
    cv = purged_walk_forward(
        sig, prices, engine,
        n_folds=cfg.validation.n_cv_folds,
        embargo=cfg.validation.embargo_days,
    )

    layer_label = "hourly" if layer == 2 else "daily"
    print(f"\nPurged Walk-Forward CV ({layer_label}, {cv.n_folds} folds):")
    print(f"  OOS Sharpe:     {cv.oos_sharpe:>8.3f} +/- {cv.oos_sharpe_std:.3f}")
    print(f"  OOS Return:     {cv.oos_return:>7.1%}")
    print(f"  OOS Max DD:     {cv.oos_max_dd:>7.1%}")
    print(f"  OOS CVaR95:     {cv.oos_cvar_95:>7.3%}")
    print(f"  OOS Log Growth: {cv.oos_expected_log_growth:>8.3f}")
    print(f"  OOS Tail Hit:   {cv.oos_tail_hit_rate:>7.1%}")
    print(f"  Fold Sharpes:   {[f'{s:.3f}' for s in cv.fold_sharpes]}")

    passed = cv.oos_sharpe >= cfg.validation.oos_sharpe_min
    print(f"\n  Gate (OOS Sharpe >= {cfg.validation.oos_sharpe_min}): {'PASS' if passed else 'FAIL'}")


def _print_paper_result(result) -> None:
    print(f"\n{'='*60}")
    print(f"Paper Trading Cycle: {result.date}")
    print(f"{'='*60}")
    print(f"  Signal Raw: {result.combined_signal:+.4f}")
    if result.strategic_signal is not None:
        print(f"  Signal L3:  {result.strategic_signal:+.4f}")
    if result.regime_adjusted_signal is not None:
        print(f"  Signal Reg: {result.regime_adjusted_signal:+.4f}")
    if result.tactical_adjusted_signal is not None:
        print(f"  Signal L2:  {result.tactical_adjusted_signal:+.4f}")
    if result.final_signal is not None:
        print(f"  Signal Fin: {result.final_signal:+.4f}")
    print(f"  Risk Scale: DD={result.dd_scale:.2f} Vol={result.vol_scale:.2f}")
    print(f"  Trades:     {len(result.fills)}")
    for f in result.fills:
        print(f"    {f.side.upper():>4} {f.qty:.6f} {f.symbol} @ ${f.price:,.2f}")
    print(f"  Portfolio:  ${result.portfolio_value:,.2f}")
    print(f"  Daily P&L:  ${result.daily_pnl:+,.2f} ({result.daily_return:+.2%})")
    if getattr(result, "profile_id", ""):
        commit = getattr(result, "profile_commit", "")
        suffix = f" ({commit[:8]})" if commit else ""
        print(f"  Profile:    {result.profile_id[:12]}{suffix}")
    print(f"  Managed:    {result.n_registry_active} active")
    print(f"  Deployed:   {result.n_deployed_alphas} alphas")
    print(f"  Shortlist:  {result.n_shortlist_candidates} candidates")
    print(f"  Selected:   {result.n_selected_alphas} alphas")
    print(f"  Signals:    {result.n_signals_evaluated} evaluated")
    if result.n_skipped_deadband > 0:
        print(f"  Skips:      deadband={result.n_skipped_deadband}")
    if result.n_skipped_min_notional > 0:
        print(f"  Skips:      min_notional={result.n_skipped_min_notional}")
    if result.n_skipped_rounded_to_zero > 0:
        print(f"  Skips:      rounded_to_zero={result.n_skipped_rounded_to_zero}")


def cmd_paper(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    interval = args.interval or cfg.forward.check_interval

    if args.replay:
        _cmd_paper_replay(args, cfg)
        return

    from alpha_os.paper.trader import Trader
    from alpha_os.pipeline.scheduler import PipelineScheduler, SchedulerConfig

    tactical = _build_tactical_trader(
        asset=args.asset,
        cfg=cfg,
        enabled=getattr(args, "tactical", False),
    )

    trader = Trader(asset=args.asset, config=cfg, tactical=tactical)

    if args.summary:
        trader.print_status()
        trader.close()
        return

    if args.once or not args.schedule:
        if tactical is not None and _needs_evolution_l2(tactical):
            l2_cfg = _build_l2_pipeline_config(cfg, 200, 30)
            print("No L2 alphas — running L2 evolution...")
            _run_l2_evolution(tactical, cfg, l2_cfg)
        result = trader.run_cycle()
        _print_paper_result(result)
        trader.print_status()
        trader.close()
        return

    def cycle():
        if tactical is not None and _needs_evolution_l2(tactical):
            l2_cfg = _build_l2_pipeline_config(cfg, 200, 30)
            print("No L2 alphas — running L2 evolution...")
            _run_l2_evolution(tactical, cfg, l2_cfg)
        result = trader.run_cycle()
        _print_paper_result(result)

    scheduler = PipelineScheduler(
        run_fn=cycle,
        config=SchedulerConfig(interval_seconds=interval),
    )
    try:
        scheduler.start()
    finally:
        trader.close()


def _cmd_paper_replay(args: argparse.Namespace, cfg) -> None:
    from alpha_os.paper.simulator import run_replay

    if not args.start or not args.end:
        print("Error: --replay requires --start and --end dates")
        sys.exit(1)

    print(f"Running historical replay: {args.start} to {args.end} ({args.asset})")
    def _print_replay_result(label: str, result) -> None:
        print(f"\nHistorical Replay [{label}]: {args.start} to {args.end}")
        print(f"  Days:       {result.n_days}")
        print(f"  {'─'*36}")
        print(f"  Initial:    ${result.initial_capital:,.2f}")
        print(f"  Final:      ${result.final_value:,.2f}")
        print(f"  Return:     {result.total_return:+.2%}")
        print(f"  Sharpe:     {result.sharpe:.3f}")
        print(f"  Max DD:     {result.max_drawdown:.2%}")
        print(f"  Trades:     {result.total_trades}")
        print(f"  Win Rate:   {result.win_rate:.1%}")
        print(f"  {'─'*36}")
        if result.best_day[0]:
            print(f"  Best Day:   {result.best_day[1]:+.2%} ({result.best_day[0]})")
        if result.worst_day[0]:
            print(f"  Worst Day:  {result.worst_day[1]:+.2%} ({result.worst_day[0]})")

    if args.sizing_mode == "compare":
        runtime_result = run_replay(
            asset=args.asset,
            config=cfg,
            start_date=args.start,
            end_date=args.end,
            sizing_mode="runtime",
        )
        raw_result = run_replay(
            asset=args.asset,
            config=cfg,
            start_date=args.start,
            end_date=args.end,
            sizing_mode="raw_mean",
        )
        _print_replay_result("runtime", runtime_result)
        _print_replay_result("raw_mean", raw_result)
        print("\nDelta (runtime - raw_mean)")
        print(f"  Final:      ${runtime_result.final_value - raw_result.final_value:+,.2f}")
        print(f"  Return:     {runtime_result.total_return - raw_result.total_return:+.2%}")
        print(f"  Sharpe:     {runtime_result.sharpe - raw_result.sharpe:+.3f}")
        print(f"  Max DD:     {runtime_result.max_drawdown - raw_result.max_drawdown:+.2%}")
        print(f"  Trades:     {runtime_result.total_trades - raw_result.total_trades:+d}")
        return

    result = run_replay(
        asset=args.asset,
        config=cfg,
        start_date=args.start,
        end_date=args.end,
        sizing_mode=args.sizing_mode,
    )
    _print_replay_result(args.sizing_mode, result)


def _build_pipeline_config(
    config: Config, pop_size: int, generations: int,
):
    """Build PipelineConfig from global Config + CLI args."""
    from alpha_os.evolution.gp import GPConfig
    from alpha_os.pipeline.runner import PipelineConfig

    return PipelineConfig(
        gp=GPConfig(
            pop_size=pop_size,
            n_generations=generations,
            max_depth=config.generation.max_depth,
            bloat_penalty=config.generation.bloat_penalty,
            depth_penalty=config.generation.depth_penalty,
            similarity_penalty=config.generation.similarity_penalty,
        ),
        oos_sharpe_min=config.validation.oos_sharpe_min,
        pbo_max=config.validation.pbo_max,
        dsr_pvalue_max=config.validation.dsr_pvalue_max,
        min_n_days=config.backtest.min_days,
        commission_pct=config.backtest.commission_pct,
        slippage_pct=config.backtest.slippage_pct,
        n_cv_folds=config.validation.n_cv_folds,
        embargo_days=config.validation.embargo_days,
        eval_window_days=config.backtest.eval_window_days,
        allow_short=config.trading.supports_short,
    )


def _run_evolution(trader, config: Config, pipeline_config) -> None:
    """Run alpha evolution using trader's data and registry."""
    from alpha_os.pipeline.runner import PipelineRunner

    logger = logging.getLogger(__name__)

    # Sync data before evolution
    try:
        trader.store.sync(trader.features)
    except Exception as exc:
        logger.warning("API sync failed before evolution — using cached data: %s", exc)

    matrix = trader.store.get_matrix(trader.features)
    if len(matrix) < config.backtest.min_days:
        logger.warning(
            "Insufficient data for evolution (%d rows, need %d)",
            len(matrix), config.backtest.min_days,
        )
        return

    matrix = matrix.fillna(0)
    available_features = [
        f for f in trader.features
        if f in matrix.columns and not (matrix[f] == 0).all()
    ]
    if trader.price_signal not in available_features:
        logger.warning("Price signal missing for evolution: %s", trader.price_signal)
        return
    data = {col: matrix[col].values for col in matrix.columns}
    prices = data[trader.price_signal]
    logger.info(
        "Evolution feature gate: %d/%d available features",
        len(available_features), len(trader.features),
    )

    runner = PipelineRunner(
        features=available_features,
        data=data,
        prices=prices,
        config=pipeline_config,
        registry=trader.registry,
    )
    result = runner.run()
    logger.info(
        "Evolution: %d generated, %d validated, %d adopted (%.1fs)",
        result.n_generated, result.n_validated, result.n_adopted, result.elapsed,
    )
    print(
        f"Evolution: {result.n_generated} generated, "
        f"{result.n_validated} validated, {result.n_adopted} adopted "
        f"({result.elapsed:.1f}s)"
    )
    del runner, data, matrix, result
    gc.collect()


def _needs_evolution_l2(tactical) -> bool:
    """Return True if L2 registry has no active alphas."""
    from alpha_os.alpha.managed_alphas import AlphaState
    active = tactical.registry.list_by_state(AlphaState.ACTIVE)
    return len(active) == 0


def _build_tactical_trader(asset: str, cfg: Config, enabled: bool):
    """Return TacticalTrader only when Layer 2 is explicitly enabled."""
    if not enabled:
        return None
    from alpha_os.paper.tactical import TacticalTrader
    return TacticalTrader(asset=asset, config=cfg)


def _build_l2_pipeline_config(config: Config, pop_size: int, generations: int):
    """Build PipelineConfig for L2 hourly alpha evolution."""
    from alpha_os.evolution.gp import GPConfig
    from alpha_os.pipeline.runner import PipelineConfig

    return PipelineConfig(
        gp=GPConfig(
            pop_size=pop_size,
            n_generations=generations,
            max_depth=config.generation.max_depth,
            bloat_penalty=config.generation.bloat_penalty,
            depth_penalty=config.generation.depth_penalty,
            similarity_penalty=config.generation.similarity_penalty,
        ),
        oos_sharpe_min=config.validation.oos_sharpe_min,
        pbo_max=config.validation.pbo_max,
        dsr_pvalue_max=config.validation.dsr_pvalue_max,
        min_n_days=168,  # 7 days * 24 hourly bars
        commission_pct=config.backtest.commission_pct,
        slippage_pct=config.backtest.slippage_pct,
        n_cv_folds=config.validation.n_cv_folds,
        embargo_days=config.validation.embargo_days,
        eval_window_days=0,
        allow_short=config.trading.supports_short,
    )


def _run_l2_evolution(tactical, config: Config, pipeline_config) -> None:
    """Run L2 hourly alpha evolution using tactical trader's data and registry."""
    from alpha_os.pipeline.runner import PipelineRunner
    from alpha_os.data.universe import price_signal

    logger = logging.getLogger(__name__)

    try:
        tactical.store.sync(tactical.features, resolution=tactical.resolution)
    except Exception as exc:
        logger.warning("L2 API sync failed before evolution — using cached data: %s", exc)

    matrix = tactical.store.get_matrix(
        tactical.features, resolution=tactical.resolution,
    )
    min_bars = pipeline_config.min_n_days
    if len(matrix) < min_bars:
        logger.warning(
            "Insufficient L2 data for evolution (%d bars, need %d)",
            len(matrix), min_bars,
        )
        return

    matrix = matrix.fillna(0)
    available_features = [
        f for f in tactical.features
        if f in matrix.columns and not (matrix[f] == 0).all()
    ]
    data = {col: matrix[col].values for col in matrix.columns}
    price_col = price_signal(tactical.asset)
    if price_col not in available_features:
        logger.warning("L2 price signal missing for evolution: %s", price_col)
        return
    prices = data[price_col]
    logger.info(
        "L2 Evolution feature gate: %d/%d available features",
        len(available_features), len(tactical.features),
    )

    runner = PipelineRunner(
        features=available_features,
        data=data,
        prices=prices,
        config=pipeline_config,
        registry=tactical.registry,
    )
    result = runner.run()
    logger.info(
        "L2 Evolution: %d generated, %d validated, %d adopted (%.1fs)",
        result.n_generated, result.n_validated, result.n_adopted, result.elapsed,
    )
    print(
        f"L2 Evolution: {result.n_generated} generated, "
        f"{result.n_validated} validated, {result.n_adopted} adopted "
        f"({result.elapsed:.1f}s)"
    )
    del runner, data, matrix, result
    gc.collect()


def _print_testnet_report(report) -> None:
    print(f"\n--- Testnet Readiness Report ({report.date}) ---")
    if getattr(report, "profile_id", ""):
        suffix = f" ({report.profile_commit[:8]})" if report.profile_commit else ""
        print(f"  Profile:        {report.profile_id[:12]}{suffix}")
    print(f"  Cycle OK:       {report.cycle_completed}")
    print(f"  Recon match:    {report.reconciliation_match}")
    print(f"  CB halted:      {report.circuit_breaker_halted}")
    print(f"  Managed:        {report.n_registry_active} active")
    print(f"  Deployed:       {report.n_deployed_alphas} alphas")
    print(f"  Shortlist:      {report.n_shortlist_candidates} candidates")
    print(f"  Selected:       {report.n_selected_alphas} alphas")
    print(f"  Signals:        {report.n_signals_evaluated} evaluated")
    if report.n_skipped_deadband > 0:
        print(f"  Skips:          deadband={report.n_skipped_deadband}")
    if report.n_skipped_min_notional > 0:
        print(f"  Skips:          min_notional={report.n_skipped_min_notional}")
    if report.n_skipped_rounded_to_zero > 0:
        print(f"  Skips:          rounded_to_zero={report.n_skipped_rounded_to_zero}")
    if report.n_fills > 0:
        print(f"  Fills:          {report.n_fills}")
        print(f"  Avg slippage:   {report.mean_slippage_bps:.1f} bps")
        print(f"  Avg latency:    {report.mean_latency_ms:.0f} ms")
    if report.has_errors:
        print("  ERRORS:")
        for e in report.error_details:
            print(f"    - {e}")
    else:
        print("  Status:         OK")


def _resolve_asset_list(args: argparse.Namespace) -> list[str]:
    """Return list of assets from --assets or --asset."""
    if args.assets:
        return [a.strip().upper() for a in args.assets.split(",")]
    return [args.asset.upper()]


def _setup_asset_context(
    asset: str, cfg, testnet: bool, capital: float,
    venue: str | None = None,
):
    """Create trader, circuit breaker, and readiness checker for one asset."""
    from alpha_os.paper.trader import Trader
    from alpha_os.risk.circuit_breaker import CircuitBreaker

    adir = asset_data_dir(asset)
    cb = CircuitBreaker.load(path=adir / "metrics" / "circuit_breaker.json")
    cfg.trading.initial_capital = capital

    resolved_venue = venue or infer_venue(asset)

    executor = None
    if resolved_venue == "binance" and is_crypto(asset):
        from alpha_os.execution.binance import BinanceExecutor
        from alpha_os.execution.optimizer import ExecutionOptimizer, ExecutionConfig
        signal_name = price_signal(asset)
        market_symbol = f"{asset}/USDT"
        symbol_map = {signal_name: market_symbol}

        client = build_signal_client_from_config(cfg.api)
        exec_cfg = ExecutionConfig(
            imbalance_threshold=cfg.execution.imbalance_threshold,
            vpin_threshold=cfg.execution.vpin_threshold,
            spread_threshold_bps=cfg.execution.spread_threshold_bps,
            max_slices=cfg.execution.max_slices,
            signal_lookback_minutes=cfg.execution.signal_lookback_minutes,
            max_signal_age_seconds=cfg.execution.max_signal_age_seconds,
        )
        optimizer = ExecutionOptimizer(client, exec_cfg)
        executor = BinanceExecutor(
            testnet=testnet, symbol_map=symbol_map, optimizer=optimizer,
            initial_capital=capital,
            cost_model=cfg.execution.to_cost_model(),
        )
    elif resolved_venue == "alpaca" and is_equity(asset):
        from alpha_os.execution.alpaca import AlpacaExecutor
        from alpha_os.execution.costs import ExecutionCostModel
        executor = AlpacaExecutor(
            paper=cfg.alpaca.paper,
            initial_capital=capital,
            cost_model=ExecutionCostModel(
                commission_pct=cfg.alpaca.commission_pct,
                modeled_slippage_pct=cfg.alpaca.modeled_slippage_pct,
            ),
            max_slippage_bps=cfg.alpaca.max_slippage_bps,
        )
    elif resolved_venue == "polymarket":
        from alpha_os.execution.polymarket import PolymarketExecutor
        from alpha_os.execution.costs import PolymarketCostModel
        executor = PolymarketExecutor(
            max_position_per_market_usd=cfg.polymarket.max_position_per_market_usd,
            initial_capital=capital,
            cost_model=PolymarketCostModel(
                maker_fee_pct=cfg.polymarket.maker_fee_pct,
                taker_fee_pct=cfg.polymarket.taker_fee_pct,
            ),
        )

    trader = Trader(asset=asset, config=cfg, executor=executor,
                    circuit_breaker=cb)

    readiness_checker = None
    if testnet and is_crypto(asset):
        from alpha_os.validation.testnet import ReadinessChecker, readiness_paths
        state_path, report_path = readiness_paths(adir)
        readiness_checker = ReadinessChecker(
            state_path=state_path,
            report_path=report_path,
            target_days=cfg.testnet.target_success_days,
            max_slippage_bps=cfg.testnet.max_acceptable_slippage_bps,
        )

    return trader, cb, readiness_checker


def _configure_trade_logging() -> None:
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"trade_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _confirm_real_trading(args: argparse.Namespace) -> bool:
    if not args.real:
        return True
    print("=" * 60)
    print("WARNING: REAL TRADING MODE — real money on Binance.")
    print("Press Ctrl+C within 5 seconds to abort...")
    print("=" * 60)
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nAborted.")
        return False
    return True


def _print_trade_runtime_banner(
    *,
    asset_list: list[str],
    interval: int,
    testnet: bool,
    profile_changes: list[str],
    cfg: Config,
) -> None:
    mode = "TESTNET" if testnet else "REAL"
    print(f"Trade runtime [{mode}]: assets={','.join(asset_list)}, interval={interval}s")
    if profile_changes:
        print("Trade profile overrides: " + ", ".join(profile_changes))
    regime_state = "on" if cfg.regime.enabled else "off"
    print(f"Trade profile: TC-weighted consensus L3, regime {regime_state}, L2 off")


def _build_trade_contexts(
    *,
    asset_list: list[str],
    cfg: Config,
    testnet: bool,
    capital: float,
    venue: str | None = None,
) -> dict[str, tuple]:
    contexts: dict[str, tuple] = {}
    for asset in asset_list:
        resolved_venue = venue or infer_venue(asset)
        trader, cb, readiness_checker = _setup_asset_context(
            asset,
            cfg,
            testnet,
            capital,
            venue=resolved_venue,
        )
        contexts[asset] = (trader, cb, readiness_checker)
        venue_label = resolved_venue
        try:
            sig = price_signal(asset)
        except KeyError:
            sig = asset.lower()
        print(f"  {asset}: {sig} via {venue_label}")
    return contexts


def _close_trade_contexts(contexts: dict[str, tuple]) -> None:
    for trader, _, _ in contexts.values():
        trader.close()


def _needs_trade_evolution(trader) -> bool:
    from alpha_os.alpha.managed_alphas import AlphaState

    active = trader.registry.list_by_state(AlphaState.ACTIVE)
    return len(active) == 0


def _run_trade_readiness_check(result, recon, cb, readiness_checker) -> None:
    if readiness_checker is None:
        return
    report = readiness_checker.validate_cycle(
        result,
        recon,
        cb,
        result.fills,
        order_failures=getattr(result, "order_failures", 0),
    )
    _print_testnet_report(report)
    readiness_checker.print_status()


def _run_trade_once(
    *,
    asset_list: list[str],
    contexts: dict[str, tuple],
    cfg: Config,
    pipeline_cfg,
) -> None:
    for asset in asset_list:
        print(f"\n{'='*40} {asset} {'='*40}")
        trader, cb, readiness_checker = contexts[asset]
        if _needs_trade_evolution(trader):
            print(f"No alphas for {asset} — running evolution...")
            _run_evolution(trader, cfg, pipeline_cfg)
        result = trader.run_cycle()
        _print_paper_result(result)
        recon = trader.reconcile()
        _run_trade_readiness_check(result, recon, cb, readiness_checker)
        trader.print_status()


def _build_trade_cycle_runner(
    *,
    asset_list: list[str],
    contexts: dict[str, tuple],
    cfg: Config,
    args: argparse.Namespace,
    pipeline_cfg,
):
    logger = logging.getLogger(__name__)
    last_evolve: dict[str, float] = {a: 0.0 for a in asset_list}
    use_lifecycle_daemon = cfg.lifecycle_daemon.enabled
    if use_lifecycle_daemon:
        logger.info("Pipeline v2: lifecycle daemon enabled — skipping inline lifecycle")

    # Cache previous cycle's raw signals for cross-asset neutralization.
    # Uses 1-cycle lag: neutralize based on prior signals.
    # Signals change slowly (daily), so lag is negligible.
    prev_raw_signals: dict[str, float] = {}

    def cycle() -> None:
        nonlocal prev_raw_signals

        # Compute cross-asset neutralization from previous cycle
        signal_overrides: dict[str, float | None] = {a: None for a in asset_list}
        if len(asset_list) > 1 and len(prev_raw_signals) == len(asset_list):
            from alpha_os.alpha.combiner import cross_asset_neutralize
            neutralized = cross_asset_neutralize(prev_raw_signals)
            signal_overrides = {a: neutralized.get(a) for a in asset_list}
            logger.info(
                "Cross-asset neutralization: raw=%s -> neutralized=%s",
                {a: f"{v:.4f}" for a, v in prev_raw_signals.items()},
                {a: f"{v:.4f}" for a, v in neutralized.items()},
            )

        for asset in asset_list:
            trader, cb, readiness_checker = contexts[asset]
            logger.info("--- %s cycle start ---", asset)
            now = time.time()
            evolve_interval = args.evolve_interval
            if evolve_interval > 0:
                if _needs_trade_evolution(trader) or (now - last_evolve[asset]) >= evolve_interval:
                    logger.info("Running alpha evolution for %s...", asset)
                    _run_evolution(trader, cfg, pipeline_cfg)
                    last_evolve[asset] = now
            result = trader.run_cycle(
                skip_lifecycle=use_lifecycle_daemon,
                signal_override=signal_overrides[asset],
            )
            # Cache raw signal for next cycle's neutralization
            prev_raw_signals[asset] = trader.last_raw_signal
            _print_paper_result(result)
            recon = trader.reconcile()
            _run_trade_readiness_check(result, recon, cb, readiness_checker)
            logger.info("--- %s cycle done ---", asset)

    return cycle


def _run_event_driven_trade(
    *,
    asset_list: list[str],
    contexts: dict[str, tuple],
    cfg: Config,
    args: argparse.Namespace,
    pipeline_cfg,
) -> None:
    import asyncio as _asyncio

    from alpha_os.paper.event_driven import EventDrivenTrader, EventTriggerConfig

    asset = asset_list[0]
    trader, _, _ = contexts[asset]
    client = build_signal_client_from_config(cfg.api)
    last_evolve: dict[str, float] = {asset: 0.0}

    ed_cfg = EventTriggerConfig(
        min_interval=cfg.event_driven.min_interval,
        max_interval=cfg.event_driven.max_interval,
        subscribe_pattern=cfg.event_driven.subscribe_pattern,
        anomaly_trigger=cfg.event_driven.anomaly_trigger,
    )
    if args.debounce is not None:
        ed_cfg.min_interval = float(args.debounce)

    print(
        f"Event-driven mode: debounce={ed_cfg.min_interval:.0f}s, "
        f"fallback={ed_cfg.max_interval:.0f}s, "
        f"pattern={ed_cfg.subscribe_pattern}"
    )

    def _pre_cycle() -> None:
        now = time.time()
        evolve_interval = args.evolve_interval
        if evolve_interval > 0:
            if _needs_trade_evolution(trader) or (now - last_evolve[asset]) >= evolve_interval:
                logging.getLogger(__name__).info("Running alpha evolution for %s...", asset)
                _run_evolution(trader, cfg, pipeline_cfg)
                last_evolve[asset] = now

    ed_trader = EventDrivenTrader(
        trader=trader,
        client=client,
        config=ed_cfg,
        pre_cycle_hook=_pre_cycle,
    )
    _asyncio.run(ed_trader.run())


def cmd_trade(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    profile_changes = _normalize_trade_config(cfg)
    interval = args.interval or cfg.forward.check_interval
    testnet = not args.real
    asset_list = _resolve_asset_list(args)
    _configure_trade_logging()

    from alpha_os.pipeline.scheduler import PipelineScheduler, SchedulerConfig

    if not _confirm_real_trading(args):
        return
    _print_trade_runtime_banner(
        asset_list=asset_list,
        interval=interval,
        testnet=testnet,
        profile_changes=profile_changes,
        cfg=cfg,
    )

    lock_path = runtime_lock_path("trade", asset_list)
    try:
        runtime_lock = hold_runtime_lock(lock_path)
        runtime_lock.__enter__()
    except RuntimeLockBusy:
        print(
            "Trade runtime already active for "
            f"{','.join(asset_list)}; skipping overlapping invocation."
        )
        return

    try:
        venue = getattr(args, "venue", None)
        contexts = _build_trade_contexts(
            asset_list=asset_list,
            cfg=cfg,
            testnet=testnet,
            capital=args.capital,
            venue=venue,
        )

        if args.summary:
            for asset in asset_list:
                print(f"\n--- {asset} ---")
                contexts[asset][0].print_status()
            _close_trade_contexts(contexts)
            return

        pipeline_cfg = _build_pipeline_config(cfg, args.pop_size, args.generations)

        if args.once or (not args.schedule and not getattr(args, "event_driven", False)):
            _run_trade_once(
                asset_list=asset_list,
                contexts=contexts,
                cfg=cfg,
                pipeline_cfg=pipeline_cfg,
            )
            _close_trade_contexts(contexts)
            return

        cycle = _build_trade_cycle_runner(
            asset_list=asset_list,
            contexts=contexts,
            cfg=cfg,
            args=args,
            pipeline_cfg=pipeline_cfg,
        )

        if getattr(args, "event_driven", False):
            try:
                _run_event_driven_trade(
                    asset_list=asset_list,
                    contexts=contexts,
                    cfg=cfg,
                    args=args,
                    pipeline_cfg=pipeline_cfg,
                )
            finally:
                _close_trade_contexts(contexts)
            return

        scheduler = PipelineScheduler(
            run_fn=cycle,
            config=SchedulerConfig(interval_seconds=interval),
        )
        try:
            scheduler.start()
        finally:
            _close_trade_contexts(contexts)
    finally:
        runtime_lock.__exit__(None, None, None)


def cmd_cross_trade(args: argparse.Namespace) -> None:
    """Run cross-sectional multi-asset trading."""
    cfg = _load_config(args.config)
    cfg.trading.initial_capital = args.capital
    interval = args.interval or cfg.forward.check_interval
    testnet = not args.real
    asset_list = [a.strip().upper() for a in args.assets.split(",")]

    _configure_trade_logging()
    logging.getLogger().info(
        "Cross-sectional trade: assets=%s, testnet=%s, capital=$%.0f",
        asset_list, testnet, args.capital,
    )

    from alpha_os.paper.cross_sectional_trader import CrossSectionalTrader
    from alpha_os.execution.binance import BinanceExecutor
    from alpha_os.data.universe import price_signal as _price_signal

    symbol_map = {}
    for asset in asset_list:
        try:
            ps = _price_signal(asset)
            symbol_map[ps] = f"{asset}/USDT"
        except KeyError:
            symbol_map[asset.lower()] = f"{asset}/USDT"

    executor = BinanceExecutor(
        testnet=testnet,
        symbol_map=symbol_map,
        initial_capital=args.capital,
        cost_model=cfg.execution.to_cost_model(),
    )

    trader = CrossSectionalTrader(
        tradeable_assets=asset_list,
        config=cfg,
        executor=executor,
    )

    if args.once:
        result = trader.run_cycle()
        print(f"Portfolio: ${result.portfolio_value:,.2f}")
        print(f"Signals: {result.neutralized_signals}")
        print(f"Fills: {len(result.fills)}")
        trader.close()
        return

    from alpha_os.pipeline.scheduler import PipelineScheduler, SchedulerConfig

    def cycle():
        trader.run_cycle()
        recon = trader.reconcile()
        for asset, r in recon.items():
            if not r["match"]:
                logging.getLogger().warning("Reconciliation mismatch for %s: %s", asset, r)

    scheduler = PipelineScheduler(cycle, SchedulerConfig(interval_seconds=interval))
    try:
        scheduler.start()
    finally:
        trader.close()


def cmd_unified_generator(args: argparse.Namespace) -> None:
    """Run the cross-asset alpha generation daemon."""
    cfg = _load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    from alpha_os.daemon.unified_generator import UnifiedAlphaGeneratorDaemon

    daemon = UnifiedAlphaGeneratorDaemon(config=cfg)
    daemon.run()


def cmd_enqueue_discovery_pool(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)

    from alpha_os.daemon.alpha_generator import enqueue_discovery_pool_candidates

    selected, inserted = enqueue_discovery_pool_candidates(
        args.asset,
        cfg,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"Discovery-pool enqueue [{mode}]: asset={args.asset}")
    print(f"  Selected: {selected}")
    print(f"  Queued:   {inserted}")


def cmd_lifecycle(args: argparse.Namespace) -> None:
    """Run daily lifecycle evaluation (Pipeline v2, oneshot)."""
    cfg = _load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    from alpha_os.daemon.lifecycle import LifecycleDaemon

    daemon = LifecycleDaemon(asset=args.asset, config=cfg)
    daemon.run()


def cmd_rebuild_managed_alphas(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)

    from alpha_os.alpha.admission_replay import rebuild_registry

    db_path = asset_data_dir(args.asset) / "alpha_registry.db"
    stats = rebuild_registry(
        db_path,
        cfg.to_lifecycle_config(),
        source=args.source,
        fail_state=args.fail_state,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )

    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(
        f"Managed-alpha rebuild [{mode}]: asset={args.asset} "
        f"source={stats.source_name} db={stats.registry_db}"
    )
    print(f"  Source rows: {stats.source_rows}")
    print(f"  Active:      {stats.active_count}")
    print(f"  Dormant:     {stats.dormant_count}")
    print(f"  Rejected:    {stats.rejected_count}")
    if stats.backup_path is not None:
        print(f"  Backup:      {stats.backup_path}")
    if not args.dry_run:
        print("  Registry rebuilt.")


def cmd_refresh_deployed_alphas(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)

    from alpha_os.alpha.deployed_alphas import refresh_deployed_alphas

    db_path = asset_data_dir(args.asset) / "alpha_registry.db"
    stats = refresh_deployed_alphas(
        db_path,
        cfg,
        asset=args.asset,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )

    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"Deployed alphas refresh [{mode}]: asset={args.asset} db={stats.registry_db}")
    print(f"  Registry active: {stats.plan.active_count}")
    print(f"  Deployed before: {stats.plan.current_count}")
    print(f"  Deployed now:    {stats.plan.deployed_count}")
    print(f"  Kept:            {len(stats.plan.kept_ids)}")
    print(f"  Added:           {len(stats.plan.added_ids)}")
    print(f"  Dropped:         {len(stats.plan.dropped_ids)}")
    print(f"  Replacements:    {stats.plan.replacement_count}")
    print(f"  Semantic dedup:  {len(stats.plan.skipped_semantic_duplicate_ids)}")
    print(f"  Signal dedup:    {len(stats.plan.skipped_signal_duplicate_ids)}")
    print(f"  Feature cap:     {len(stats.plan.skipped_feature_cap_ids)}")
    if stats.backup_path is not None:
        print(f"  Backup:          {stats.backup_path}")


def cmd_prune_managed_alpha_duplicates(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)

    from alpha_os.alpha.deployed_alphas import prune_registry_active_duplicates

    db_path = asset_data_dir(args.asset) / "alpha_registry.db"
    stats = prune_registry_active_duplicates(
        db_path,
        cfg,
        asset=args.asset,
        dry_run=args.dry_run,
        backup=not args.no_backup,
        refresh_deployed=not args.no_refresh_deployed,
    )

    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"Managed-alpha duplicate prune [{mode}]: asset={args.asset} db={stats.registry_db}")
    print(f"  Active before:    {stats.plan.active_count}")
    print(f"  Active kept:      {stats.plan.kept_count}")
    print(f"  Demoted:          {stats.plan.demoted_count}")
    print(f"  Deployed before:  {stats.plan.current_deployed_count}")
    print(f"  Deployed touched: {stats.plan.touched_deployed_count}")
    print(f"  Semantic dedup:   {len(stats.plan.skipped_semantic_duplicate_ids)}")
    print(f"  Signal dedup:     {len(stats.plan.skipped_signal_duplicate_ids)}")
    if stats.deployed_refresh is not None:
        print(
            f"  Deployed refresh: {stats.deployed_refresh.plan.current_count}"
            f" -> {stats.deployed_refresh.plan.deployed_count}"
        )
    if stats.backup_path is not None:
        print(f"  Backup:           {stats.backup_path}")


def cmd_replay_experiment(args: argparse.Namespace) -> None:
    from alpha_os.experiments.replay import (
        ReplayExperimentSpec,
        parse_override_assignment,
        run_replay_experiment,
    )

    overrides = dict(parse_override_assignment(raw) for raw in args.set)
    run = run_replay_experiment(
        ReplayExperimentSpec(
            name=args.name,
            asset=args.asset,
            start_date=args.start,
            end_date=args.end,
            config_path=Path(args.config) if args.config else None,
            managed_alpha_mode=args.managed_alpha_mode,
            admission_source=args.source,
            fail_state=args.fail_state,
            deployment_mode=args.deployment_mode,
            sizing_mode=args.sizing_mode,
            overrides=overrides,
            notes=args.notes,
        )
    )

    result = run.payload["result"]
    profile = run.payload.get("runtime_profile", {})
    profile_id = profile.get("profile_id", "")
    profile_commit = profile.get("git_commit", "")
    print(f"Replay experiment: {run.experiment_id}")
    print(f"  Detail:   {run.detail_path}")
    print(f"  Index:    {run.index_path}")
    if profile_id:
        suffix = f" ({profile_commit[:8]})" if profile_commit else ""
        print(f"  Profile:  {profile_id[:12]}{suffix}")
    print(f"  Deployment: {run.payload['deployment']['mode']}")
    print(f"  Final:    ${result['final_value']:,.2f}")
    print(f"  Return:   {result['total_return']:+.2%}")
    print(f"  Sharpe:   {result['sharpe']:.3f}")
    print(f"  Max DD:   {result['max_drawdown']:.2%}")
    print(f"  Trades:   {result['total_trades']}")


def cmd_replay_matrix(args: argparse.Namespace) -> None:
    from alpha_os.experiments.matrix import load_replay_matrix, run_replay_matrix

    matrix = load_replay_matrix(Path(args.manifest))
    runs = run_replay_matrix(matrix, max_workers=args.max_workers)

    print(f"Replay matrix: {args.manifest}")
    profile_ids: set[str] = set()
    for run in runs:
        result = run.payload["result"]
        profile_id = run.payload.get("runtime_profile", {}).get("profile_id", "")
        if profile_id:
            profile_ids.add(profile_id)
        profile_text = f" profile={profile_id[:12]}" if profile_id else ""
        print(
            f"  - {run.payload['name']}: "
            f"return={result['total_return']:+.2%} "
            f"sharpe={result['sharpe']:.3f} "
            f"dd={result['max_drawdown']:.2%} "
            f"trades={result['total_trades']} "
            f"{profile_text} "
            f"detail={run.detail_path}"
        )
    print(f"  Profiles: {len(profile_ids)} unique across {len(runs)} runs")


def cmd_admission_daemon(args: argparse.Namespace) -> None:
    """Run the candidate admission daemon (Pipeline v2)."""
    cfg = _load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    from alpha_os.daemon.admission import AdmissionDaemon

    daemon = AdmissionDaemon(asset=args.asset, config=cfg)
    daemon.run()


def cmd_prune_stale_candidates(args: argparse.Namespace) -> None:
    from alpha_os.alpha.admission_queue import prune_stale_pending_candidates

    stats = prune_stale_pending_candidates(
        args.asset,
        max_age_days=args.max_age_days,
        dry_run=args.dry_run,
    )
    print(f"Stale candidates: asset={stats.asset}")
    print(f"  Max age days: {stats.max_age_days}")
    print(f"  Selected:     {stats.selected_count}")
    print(f"  Pruned:       {stats.pruned_count}")
    if args.dry_run:
        print("  Mode:         dry-run")


def cmd_testnet_readiness(args: argparse.Namespace) -> None:
    from alpha_os.validation.testnet import ReadinessChecker, readiness_paths

    cfg = _load_runtime_observation_config(getattr(args, "config", None))
    adir = asset_data_dir(args.asset)
    state_path, report_path = readiness_paths(adir)

    readiness_checker = ReadinessChecker(
        state_path=state_path,
        report_path=report_path,
        target_days=cfg.testnet.target_success_days,
        max_slippage_bps=cfg.testnet.max_acceptable_slippage_bps,
    )

    if args.reset:
        readiness_checker._state.consecutive_success_days = 0
        readiness_checker._state.passed = False
        readiness_checker._save_state()
        print("Reset consecutive success counter to 0.")
        return

    readiness_checker.print_status()

    if args.slippage or args.latency:
        from alpha_os.paper.tracker import PaperPortfolioTracker
        tracker = PaperPortfolioTracker(db_path=adir / "paper_trading.db")
        if args.slippage:
            stats = tracker.get_slippage_stats()
            print(f"\nSlippage Distribution ({stats['count']} fills)")
            print(f"  Mean:  {stats['mean_bps']:.1f} bps")
            print(f"  P50:   {stats['p50_bps']:.1f} bps")
            print(f"  P95:   {stats['p95_bps']:.1f} bps")
            print(f"  Max:   {stats['max_bps']:.1f} bps")
        if args.latency:
            stats = tracker.get_latency_stats()
            print(f"\nFill Latency Distribution ({stats['count']} fills)")
            print(f"  Mean:  {stats['mean_ms']:.0f} ms")
            print(f"  P50:   {stats['p50_ms']:.0f} ms")
            print(f"  P95:   {stats['p95_ms']:.0f} ms")
            print(f"  Max:   {stats['max_ms']:.0f} ms")
        tracker.close()

    if args.reports:
        if not report_path.exists():
            print("\nNo reports yet.")
            return
        print("\nDaily Reports:")
        for line in report_path.read_text().splitlines():
            r = json.loads(line)
            status = "OK" if not r["has_errors"] else "ERROR"
            print(
                f"  {r['date']} [{status}] PV=${r['portfolio_value']:,.2f} "
                f"PnL=${r['daily_pnl']:+,.2f} fills={r['n_fills']}"
            )


def _load_latest_report(report_path: Path) -> dict | None:
    if not report_path.exists():
        return None
    last: dict | None = None
    for line in report_path.read_text().splitlines():
        line = line.strip()
        if line:
            last = json.loads(line)
    return last


def _managed_alpha_status(adir: Path) -> dict[str, int]:
    from alpha_os.alpha.managed_alphas import ManagedAlphaStore, AlphaState

    registry = ManagedAlphaStore(adir / "alpha_registry.db")
    try:
        return {
            "active": registry.count(AlphaState.ACTIVE),
            "dormant": registry.count(AlphaState.DORMANT),
            "rejected": registry.count(AlphaState.REJECTED),
            "deployed": registry.count_deployed_alphas(),
        }
    finally:
        registry.close()


def _runtime_observation_findings(
    latest: dict | None,
    managed_alphas: dict[str, int],
) -> list[str]:
    findings: list[str] = []
    if latest is None:
        findings.append("no readiness reports yet")
        return findings

    if latest.get("has_errors"):
        findings.append("latest cycle has runtime errors")
    if latest.get("n_fills", 0) == 0:
        findings.append("latest cycle had zero fills")
    if latest.get("n_skipped_deadband", 0) > 0 and latest.get("n_fills", 0) == 0:
        findings.append("deadband skipped the latest cycle")
    if latest.get("n_order_failures", 0) > 0:
        findings.append("latest cycle had order failures")
    if not latest.get("reconciliation_match", False):
        findings.append("latest cycle failed reconciliation")
    if (
        latest.get("n_registry_active", managed_alphas["active"])
        != managed_alphas["active"]
    ):
        findings.append("latest report is older than current managed-alpha state")
    return findings


def _latest_deployed_count(latest: dict | None) -> int:
    if latest is None:
        return 0
    return int(
        latest.get(
            "n_deployed_alphas",
            latest.get("n_universe_deployed", 0),
        )
    )


def _current_runtime_profile(cfg, adir: Path, asset: str):
    from alpha_os.alpha.managed_alphas import ManagedAlphaStore

    registry = ManagedAlphaStore(adir / "alpha_registry.db")
    try:
        deployed_ids = [record.alpha_id for record in registry.list_deployed_alphas()]
    finally:
        registry.close()
    profile = build_runtime_profile(
        asset=asset,
        config=cfg,
        deployed_alpha_ids=deployed_ids,
    )
    return profile


def cmd_runtime_status(args: argparse.Namespace) -> None:
    from alpha_os.validation.testnet import ReadinessChecker, readiness_paths

    cfg = _load_runtime_observation_config(getattr(args, "config", None))
    adir = asset_data_dir(args.asset)
    state_path, report_path = readiness_paths(adir)
    readiness_checker = ReadinessChecker(
        state_path=state_path,
        report_path=report_path,
        target_days=cfg.testnet.target_success_days,
        max_slippage_bps=cfg.testnet.max_acceptable_slippage_bps,
    )
    state = readiness_checker.state
    latest = _load_latest_report(report_path)
    managed_alphas = _managed_alpha_status(adir)
    current_profile = _current_runtime_profile(cfg, adir, args.asset)
    findings = _runtime_observation_findings(latest, managed_alphas)

    print(f"Runtime Status ({args.asset.upper()})")
    print(
        f"  Readiness: {state.consecutive_success_days}/{state.target_days} "
        f"days, total={state.total_days_run}, passed={state.passed}"
    )
    print(
        f"  Last Run:  {state.last_run_date or 'N/A'} "
        f"(last_success={state.last_success_date or 'N/A'})"
    )
    print(
        f"  Managed:   active={managed_alphas['active']} dormant={managed_alphas['dormant']} "
        f"rejected={managed_alphas['rejected']} deployed={managed_alphas['deployed']}"
    )
    commit_suffix = f" ({current_profile.git_commit[:8]})" if current_profile.git_commit else ""
    print(f"  Profile:   current={current_profile.profile_id[:12]}{commit_suffix}")

    if latest is None:
        print("  Latest:    no readiness reports yet")
        return

    status = "ERROR" if latest.get("has_errors") else "OK"
    print(
        f"  Latest:    {latest['date']} [{status}] PV=${latest['portfolio_value']:,.2f} "
        f"PnL=${latest['daily_pnl']:+,.2f} fills={latest['n_fills']}"
    )
    latest_profile_id = latest.get("profile_id", "")
    latest_profile_commit = latest.get("profile_commit", "")
    if latest_profile_id:
        latest_suffix = f" ({latest_profile_commit[:8]})" if latest_profile_commit else ""
        print(f"  Profile:   latest={latest_profile_id[:12]}{latest_suffix}")
        latest_config_id = latest.get("profile_config_id", "")
        latest_deployed_set_id = latest.get("profile_deployed_set_id", "")
        if latest_config_id or latest_deployed_set_id:
            print(
                "  ProfileIDs: "
                f"config={latest_config_id[:12] or 'n/a'} "
                f"deployed={latest_deployed_set_id[:12] or 'n/a'}"
            )
            print(
                "  CurrentIDs: "
                f"config={current_profile.config_id[:12]} "
                f"deployed={current_profile.deployed_set_id[:12]}"
            )
    print(
        f"  Selection: managed_active={latest['n_registry_active']} "
        f"deployed={_latest_deployed_count(latest)} "
        f"selected={latest['n_selected_alphas']}"
    )
    print(
        "  Skips:     "
        f"deadband={latest['n_skipped_deadband']} "
        f"min_notional={latest['n_skipped_min_notional']} "
        f"rounded_to_zero={latest['n_skipped_rounded_to_zero']}"
    )
    print(
        f"  Health:    reconciliation={latest['reconciliation_match']} "
        f"order_failures={latest['n_order_failures']} "
        f"halted={latest['circuit_breaker_halted']}"
    )
    verdict = "pending"
    if any(
        finding in findings
        for finding in (
            "latest cycle has runtime errors",
            "latest cycle had order failures",
            "latest cycle failed reconciliation",
        )
    ):
        verdict = "attention"
    print(f"  Observe:   {verdict}")
    for finding in findings:
        print(f"    - {finding}")
    if latest_profile_id and latest_profile_id != current_profile.profile_id:
        print("    - latest report was recorded under a different runtime profile")
        latest_config_id = latest.get("profile_config_id", "")
        latest_deployed_set_id = latest.get("profile_deployed_set_id", "")
        if latest_config_id and latest_config_id != current_profile.config_id:
            print("    - config fingerprint differs between current and latest")
        if latest_deployed_set_id and latest_deployed_set_id != current_profile.deployed_set_id:
            print("    - deployed alpha set fingerprint differs between current and latest")
    if latest["n_registry_active"] != managed_alphas["active"]:
        print(
            "  Note:      managed-alpha DB count differs from latest readiness report; "
            "the next trade cycle will refresh the report."
        )


def cmd_alpha_funnel(args: argparse.Namespace) -> None:
    from alpha_os.alpha.funnel import load_funnel_summary

    summary = load_funnel_summary(args.asset)
    print(f"Alpha Funnel ({summary.asset.upper()})")
    print(f"  Discovery: pool={summary.discovery_pool_entries}")
    print(
        "  Candidates:"
        f" total={summary.candidate_total}"
        f" pending={summary.candidate_pending}"
        f" validating={summary.candidate_validating}"
        f" adopted={summary.candidate_adopted}"
        f" rejected={summary.candidate_rejected}"
    )
    print(
        "  Enqueued:"
        f" total={summary.enqueued_total}"
        f" manual={summary.enqueued_manual}"
    )
    print(
        "  Managed:"
        f" candidate={summary.managed_candidate}"
        f" active={summary.managed_active}"
        f" dormant={summary.managed_dormant}"
        f" rejected={summary.managed_rejected}"
    )
    print(f"  Deployed: total={summary.deployed_total}")
    if summary.reject_axes:
        axes = " ".join(f"{axis}={count}" for axis, count in summary.reject_axes)
        print(f"  Reject Axes: {axes}")
    if summary.reject_reasons:
        print("  Top Rejects:")
        for reason, count in summary.reject_reasons:
            print(f"    - {count}x {reason}")
    if summary.source_summaries:
        print("  By Source:")
        for row in summary.source_summaries:
            print(
                "    "
                f"{row.source}: "
                f"total={row.total} "
                f"pending={row.pending} "
                f"validating={row.validating} "
                f"adopted={row.adopted} "
                f"rejected={row.rejected}"
            )
            if row.reject_axes:
                axes = " ".join(f"{axis}={count}" for axis, count in row.reject_axes)
                print(f"      axes: {axes}")
            for reason, count in row.top_reject_reasons:
                print(f"      - {count}x {reason}")


def cmd_evaluate_expression(args: argparse.Namespace) -> None:
    """Evaluate expression with multi-horizon IC across eval universe."""
    from pathlib import Path
    from alpha_os.alpha.cross_asset import evaluate_cross_asset_multi_horizon, DEFAULT_HORIZONS
    from alpha_os.config import Config, DATA_DIR
    from alpha_os.data.store import DataStore
    from alpha_os.data.signal_client import build_signal_client_from_config
    from alpha_os.data.universe import init_universe, load_daily_signals
    from alpha_os.data.eval_universe import load_cached_eval_universe

    cfg = Config.load(args.config)
    client = build_signal_client_from_config(cfg.api)
    price_signals = init_universe(client)
    all_signals = load_daily_signals(client)

    # Load cached eval universe
    eval_assets = load_cached_eval_universe()
    if not eval_assets:
        print("No cached eval universe. Run unified-generator first.")
        return

    # Load data: eval universe prices + features referenced by expression
    from alpha_os.dsl import parse as dsl_parse
    from alpha_os.alpha.expression_identity import expression_feature_names
    expr_features = expression_feature_names(args.expr)
    db_path = DATA_DIR / "alpha_cache.db"
    store = DataStore(db_path, client)
    needed = sorted(set(eval_assets) | expr_features | set(all_signals[:50]))
    matrix = store.get_matrix(needed)
    store.close()
    eval_set = set(eval_assets)
    for col in matrix.columns:
        if col not in eval_set:
            matrix[col] = matrix[col].fillna(0)
    data = {col: matrix[col].values for col in matrix.columns}

    print(f"Expression: {args.expr}")
    print(f"Eval universe: {len(eval_assets)} assets")
    print(f"Data: {len(matrix)} rows, {len(data)} signals")
    print()

    result = evaluate_cross_asset_multi_horizon(
        args.expr, data, eval_assets,
        horizons=DEFAULT_HORIZONS,
        fitness_metric="ic",
        benchmark_assets=cfg.backtest.benchmark_assets,
    )

    print("Horizon  Mean IC")
    print("-" * 25)
    for h in sorted(result.fitness_by_horizon):
        ic = result.fitness_by_horizon[h]
        marker = " <-- best" if h == result.best_horizon else ""
        print(f"  {h:2d}d    {ic:+.4f}{marker}")
    print()
    print(f"Best: horizon={result.best_horizon}d, IC={result.best_fitness:+.4f}")

    if result.per_asset:
        print(f"\nPer-asset IC (top 10):")
        sorted_assets = sorted(result.per_asset.items(), key=lambda x: x[1], reverse=True)
        for asset, ic in sorted_assets[:10]:
            print(f"  {asset:30s} IC={ic:+.4f}")


def cmd_submit_expression(args: argparse.Namespace) -> None:
    """Submit expression to admission queue."""
    from pathlib import Path
    from alpha_os.alpha.cross_asset import evaluate_cross_asset_multi_horizon, DEFAULT_HORIZONS
    from alpha_os.alpha.managed_alphas import ManagedAlphaStore
    from alpha_os.config import Config, DATA_DIR, asset_data_dir
    from alpha_os.data.store import DataStore
    from alpha_os.data.signal_client import build_signal_client_from_config
    from alpha_os.data.universe import init_universe, load_daily_signals
    from alpha_os.data.eval_universe import load_cached_eval_universe

    cfg = Config.load(args.config)
    client = build_signal_client_from_config(cfg.api)
    price_signals = init_universe(client)
    all_signals = load_daily_signals(client)

    eval_assets = load_cached_eval_universe()
    if not eval_assets:
        print("No cached eval universe. Run unified-generator first.")
        return

    from alpha_os.alpha.expression_identity import expression_feature_names
    expr_features = expression_feature_names(args.expr)
    db_path = DATA_DIR / "alpha_cache.db"
    store = DataStore(db_path, client)
    needed = sorted(set(eval_assets) | expr_features | set(all_signals[:50]))
    matrix = store.get_matrix(needed)
    store.close()
    eval_set = set(eval_assets)
    for col in matrix.columns:
        if col not in eval_set:
            matrix[col] = matrix[col].fillna(0)
    data = {col: matrix[col].values for col in matrix.columns}

    # Evaluate
    result = evaluate_cross_asset_multi_horizon(
        args.expr, data, eval_assets,
        horizons=DEFAULT_HORIZONS,
        fitness_metric="ic",
        benchmark_assets=cfg.backtest.benchmark_assets,
    )

    print(f"Expression: {args.expr}")
    print(f"Best horizon: {result.best_horizon}d, IC: {result.best_fitness:+.4f}")

    if result.best_fitness <= 0:
        print("IC <= 0. Not submitting.")
        return

    # Submit to admission queue
    registry = ManagedAlphaStore(asset_data_dir(args.asset) / "alpha_registry.db")
    try:
        inserted = registry.queue_candidate_expressions(
            [args.expr],
            source="human",
            behavior_json={
                "source": "human",
                "best_horizon": result.best_horizon,
                "fitness_by_horizon": {str(k): v for k, v in result.fitness_by_horizon.items()},
            },
        )
    finally:
        registry.close()

    if inserted > 0:
        print(f"Submitted to admission queue (source=human, horizon={result.best_horizon}d)")
    else:
        print("Already in queue (duplicate)")


def cmd_produce_classical(args: argparse.Namespace) -> None:
    """Run classical indicator producer → prediction store."""
    from alpha_os.config import Config
    from alpha_os.predictions.classical_producer import produce_classical_predictions

    cfg = Config.load(args.config)
    n = produce_classical_predictions(cfg)
    print(f"Wrote {n} classical predictions to store")


def cmd_produce_predictions(args: argparse.Namespace) -> None:
    """Run registry producer: evaluate active alphas → prediction store."""
    from alpha_os.config import Config
    from alpha_os.predictions.registry_producer import produce_daily_predictions

    cfg = Config.load(args.config)
    n = produce_daily_predictions(args.asset, cfg)
    print(f"Wrote {n} predictions to store")


def cmd_seed_handcrafted(args: argparse.Namespace) -> None:
    from alpha_os.alpha.handcrafted import (
        get_handcrafted_expressions,
        list_handcrafted_sets,
    )
    from alpha_os.alpha.managed_alphas import ManagedAlphaStore

    asset = args.asset.upper()
    available_sets = list_handcrafted_sets(asset)
    if args.list:
        print(f"Hand-crafted sets ({asset})")
        if not available_sets:
            print("  none")
            return
        for name in available_sets:
            print(f"  - {name}")
        return

    try:
        expressions = get_handcrafted_expressions(asset, args.alpha_set)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Seed hand-crafted set: asset={asset} set={args.alpha_set} "
        f"count={len(expressions)}"
    )
    for expression in expressions:
        print(f"  {expression}")
    if args.dry_run:
        return

    registry = ManagedAlphaStore(asset_data_dir(asset) / "alpha_registry.db")
    try:
        inserted = registry.queue_candidate_expressions(
            expressions,
            source=f"manual_{asset.lower()}_{args.alpha_set}",
            behavior_json={"source": "handcrafted", "set": args.alpha_set, "asset": asset},
        )
    finally:
        registry.close()
    print(f"Queued: {inserted} new candidates")


def cmd_analyze_diversity(args: argparse.Namespace) -> None:
    from alpha_os.alpha.diversity import analyze_diversity
    from alpha_os.alpha.managed_alphas import ManagedAlphaStore, AlphaState

    cfg = _load_runtime_observation_config(getattr(args, "config", None))
    features = build_feature_list(args.asset)
    data, n_days = _real_data(features, cfg, eval_window=max(args.lookback, 0))

    registry = ManagedAlphaStore(asset_data_dir(args.asset) / "alpha_registry.db")
    try:
        if args.scope == "deployed":
            records = registry.list_deployed_alphas()
            if args.limit > 0:
                records = records[:args.limit]
        else:
            if args.limit > 0:
                records = registry.top(
                    args.limit,
                    state=AlphaState.ACTIVE,
                    metric=args.metric,
                )
            else:
                records = registry.list_active()
    finally:
        registry.close()

    report = analyze_diversity(
        records,
        data,
        n_days,
        lookback=args.lookback,
        top_pairs=args.top_pairs,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    summary = report.summary
    print(
        f"Diversity Analysis ({args.asset.upper()} {args.scope}, "
        f"analyzed={summary.n_analyzed}/{summary.n_records}, lookback={summary.lookback})"
    )
    print(
        "  Summary:   "
        f"signal_div={summary.signal_diversity:.3f} "
        f"feature_div={summary.feature_diversity:.3f} "
        f"struct_div={summary.structure_diversity:.3f} "
        f"composite_div={summary.composite_diversity:.3f}"
    )
    print(
        "  Overlap:   "
        f"signal={summary.mean_abs_signal_correlation:.3f} "
        f"feature={summary.mean_feature_overlap:.3f} "
        f"struct={summary.mean_structure_overlap:.3f} "
        f"composite={summary.mean_composite_similarity:.3f}"
    )
    print(
        "  Inputs:    "
        f"unique={summary.n_unique_features} "
        f"input_div={summary.input_diversity:.3f} "
        f"input_corr={summary.mean_abs_input_correlation:.3f}"
    )
    if summary.family_counts:
        family_blob = ", ".join(
            f"{name}={count}" for name, count in summary.family_counts.items()
        )
        print(f"  Families:  {family_blob}")
    if summary.feature_usage_counts:
        top_usage = list(summary.feature_usage_counts.items())[:10]
        usage_blob = ", ".join(f"{name}={count}" for name, count in top_usage)
        print(f"  Features:  {usage_blob}")
    if report.skipped_alpha_ids:
        skipped_preview = ", ".join(report.skipped_alpha_ids[:5])
        suffix = "" if len(report.skipped_alpha_ids) <= 5 else ", ..."
        print(
            f"  Skipped:   {len(report.skipped_alpha_ids)} "
            f"({skipped_preview}{suffix})"
        )

    if report.top_redundant_pairs:
        print("  Top Pairs:")
        for pair in report.top_redundant_pairs:
            print(
                "    "
                f"{pair.alpha_id_a} <-> {pair.alpha_id_b} "
                f"comp={pair.composite_similarity:.3f} "
                f"sig={pair.abs_signal_correlation:.3f} "
                f"feat={pair.feature_overlap:.3f} "
                f"struct={pair.structure_overlap:.3f}"
            )

    if report.top_input_pairs:
        print("  Top Inputs:")
        for pair in report.top_input_pairs:
            print(
                "    "
                f"{pair.feature_a} <-> {pair.feature_b} "
                f"corr={pair.abs_input_correlation:.3f}"
            )

    if report.rows:
        print("  Most Redundant:")
        for row in report.rows[: min(10, len(report.rows))]:
            families = ",".join(row.feature_families) or "-"
            print(
                "    "
                f"{row.alpha_id} comp={row.avg_composite_similarity:.3f} "
                f"sig={row.avg_abs_signal_correlation:.3f} "
                f"feat={row.avg_feature_overlap:.3f} "
                f"struct={row.avg_structure_overlap:.3f} "
                f"nodes={row.node_count} families={families}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "evolve":
        cmd_evolve(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "paper":
        cmd_paper(args)
    elif args.command == "trade":
        cmd_trade(args)
    elif args.command == "cross-trade":
        cmd_cross_trade(args)
    elif args.command == "unified-generator":
        cmd_unified_generator(args)
    elif args.command == "enqueue-discovery-pool":
        cmd_enqueue_discovery_pool(args)
    elif args.command == "admission-daemon":
        cmd_admission_daemon(args)
    elif args.command == "prune-stale-candidates":
        cmd_prune_stale_candidates(args)
    elif args.command == "lifecycle":
        cmd_lifecycle(args)
    elif args.command == "rebuild-managed-alphas":
        cmd_rebuild_managed_alphas(args)
    elif args.command == "refresh-deployed-alphas":
        cmd_refresh_deployed_alphas(args)
    elif args.command == "prune-managed-alpha-duplicates":
        cmd_prune_managed_alpha_duplicates(args)
    elif args.command == "replay-experiment":
        cmd_replay_experiment(args)
    elif args.command == "replay-matrix":
        cmd_replay_matrix(args)
    elif args.command == "testnet-readiness":
        cmd_testnet_readiness(args)
    elif args.command == "runtime-status":
        cmd_runtime_status(args)
    elif args.command == "alpha-funnel":
        cmd_alpha_funnel(args)
    elif args.command == "seed-handcrafted":
        cmd_seed_handcrafted(args)
    elif args.command == "analyze-diversity":
        cmd_analyze_diversity(args)
    elif args.command == "evaluate":
        cmd_evaluate_expression(args)
    elif args.command == "submit":
        cmd_submit_expression(args)
    elif args.command == "produce-predictions":
        cmd_produce_predictions(args)
    elif args.command == "produce-classical":
        cmd_produce_classical(args)
