"""CLI for alpha-os: generate, backtest, validate alpha factors."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from datetime import date

import numpy as np

from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.config import Config, DATA_DIR, asset_data_dir
from alpha_os.alpha.evaluator import FAILED_FITNESS
from alpha_os.data.universe import is_crypto, price_signal, build_feature_list, build_hourly_feature_list
from alpha_os.dsl import parse, to_string
from alpha_os.dsl.generator import AlphaGenerator
from alpha_os.evolution.archive import AlphaArchive
from alpha_os.evolution.behavior import compute_behavior
from alpha_os.evolution.gp import GPConfig, GPEvolver
from alpha_os.validation.purged_cv import purged_walk_forward


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
    evo.add_argument("--live", action="store_true",
                    help="(deprecated — real data is now the default)")
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
    val.add_argument("--live", action="store_true",
                    help="(deprecated — real data is now the default)")
    val.add_argument("--config", type=str, default=None)

    # forward
    fwd = sub.add_parser("forward", help="Forward-test adopted alphas on new data")
    fwd.add_argument("--once", action="store_true", help="Run one cycle and exit")
    fwd.add_argument("--schedule", action="store_true", help="Run on interval")
    fwd.add_argument("--summary", action="store_true", help="Print summary and exit")
    fwd.add_argument("--interval", type=int, default=None,
                     help="Override check_interval in seconds (default: from config)")
    fwd.add_argument("--asset", type=str, default="NVDA")
    fwd.add_argument("--config", type=str, default=None)

    # paper
    ppr = sub.add_parser("paper", help="Paper trade with adopted alphas")
    ppr.add_argument("--once", action="store_true", help="Run one cycle and exit")
    ppr.add_argument("--schedule", action="store_true", help="Run on interval")
    ppr.add_argument("--summary", action="store_true", help="Print summary and exit")
    ppr.add_argument("--interval", type=int, default=None,
                     help="Override check_interval in seconds (default: from config)")
    ppr.add_argument("--backfill", action="store_true",
                     help="Run historical simulation over date range")
    ppr.add_argument("--start", type=str, default=None,
                     help="Start date for backfill (ISO format, e.g. 2025-06-01)")
    ppr.add_argument("--end", type=str, default=None,
                     help="End date for backfill (ISO format, e.g. 2026-02-25)")
    ppr.add_argument("--asset", type=str, default="BTC")
    ppr.add_argument("--config", type=str, default=None)

    # live
    liv = sub.add_parser("live", help="Live trade on Binance (testnet by default)")
    liv.add_argument("--once", action="store_true", help="Run one cycle and exit")
    liv.add_argument("--schedule", action="store_true", help="Run on interval")
    liv.add_argument("--summary", action="store_true", help="Print summary and exit")
    liv.add_argument("--interval", type=int, default=None,
                     help="Override check_interval in seconds (default: from config)")
    liv.add_argument("--real", action="store_true",
                     help="Use real Binance (default is testnet)")
    liv.add_argument("--capital", type=float, default=10000.0,
                     help="Initial capital for tracking (default: 10000)")
    liv.add_argument("--asset", type=str, default="BTC")
    liv.add_argument("--assets", type=str, default=None,
                     help="Comma-separated asset list (e.g. BTC,ETH,SOL)")
    liv.add_argument("--config", type=str, default=None)
    liv.add_argument("--evolve-interval", type=int, default=86400,
                     help="Alpha evolution interval in seconds (default: 86400=24h, 0=disable)")
    liv.add_argument("--pop-size", type=int, default=200,
                     help="GP population size for evolution")
    liv.add_argument("--generations", type=int, default=30,
                     help="GP generations per evolution cycle")

    # validate-testnet
    vt = sub.add_parser("validate-testnet", help="Check Phase 4 testnet validation status")
    vt.add_argument("--reports", action="store_true", help="Show all daily reports")
    vt.add_argument("--slippage", action="store_true", help="Show slippage distribution")
    vt.add_argument("--latency", action="store_true", help="Show fill latency distribution")
    vt.add_argument("--reset", action="store_true", help="Reset consecutive day counter")
    vt.add_argument("--asset", type=str, default="BTC")
    vt.add_argument("--config", type=str, default=None)

    return parser


def _load_config(config_path: str | None) -> Config:
    from pathlib import Path
    if config_path:
        return Config.load(Path(config_path))
    return Config.load()


def _make_features(asset: str) -> list[str]:
    """Feature names available for alpha generation."""
    return build_feature_list(asset)


def _warn_deprecated_live(args: argparse.Namespace) -> None:
    if getattr(args, "live", False):
        warnings.warn(
            "--live is deprecated. Real data is now the default. "
            "Use --synthetic for synthetic data.",
            DeprecationWarning,
            stacklevel=3,
        )


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

    from signal_noise.client import SignalClient
    client = SignalClient(
        base_url=config.api.base_url,
        timeout=config.api.timeout,
    )
    store = DataStore(db_path, client)

    # Sync from REST API (incremental, best-effort)
    try:
        if client.health():
            print(f"Syncing from {config.api.base_url} (resolution={resolution}) ...")
            store.sync(features, resolution=resolution)
        else:
            print(f"API unavailable at {config.api.base_url} — using cache")
    except Exception:
        print("API sync failed — using cache")

    matrix = store.get_matrix(features, resolution=resolution)

    # Only require price signal (first feature) to be present
    price_col = features[0]
    if price_col in matrix.columns:
        matrix = matrix[matrix[price_col].notna()]

    # Fill remaining NaN: ffill already done in get_matrix, bfill + 0 for leading
    matrix = matrix.bfill().fillna(0)

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
    _warn_deprecated_live(args)
    cfg = _load_config(args.config)
    features = _make_features(args.asset)
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
        CostModel(cfg.backtest.commission_pct, cfg.backtest.slippage_pct)
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
    print(f"{'Rank':>4}  {'Sharpe':>8}  {'Return':>8}  {'MaxDD':>8}  {'Turnover':>8}  Expression")
    print("-" * 90)
    for i, (expr, res) in enumerate(ranked[: args.top]):
        print(
            f"{i + 1:>4}  {res.sharpe:>8.3f}  {res.annual_return:>7.1%}  "
            f"{res.max_drawdown:>7.1%}  {res.turnover:>8.3f}  {to_string(expr)}"
        )


def cmd_evolve(args: argparse.Namespace) -> None:
    _warn_deprecated_live(args)
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
        CostModel(cfg.backtest.commission_pct, cfg.backtest.slippage_pct)
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
    )
    evolver = GPEvolver(features, evaluate_fn, config=gp_cfg, seed=args.seed)

    t0 = time.perf_counter()
    results = evolver.run()
    evolve_time = time.perf_counter() - t0

    # Fill MAP-Elites archive
    archive = AlphaArchive()
    live_signals: list[np.ndarray] = []
    added = 0
    for expr, fitness in results:
        try:
            sig = expr.evaluate(data)
            sig = np.nan_to_num(np.asarray(sig, dtype=float), nan=0.0)
            if sig.ndim == 0:
                sig = np.full(n_days, float(sig))
            behavior = compute_behavior(sig, expr, live_signals)
            if archive.add(expr, fitness, behavior):
                added += 1
                live_signals.append(sig)
        except Exception:
            continue

    total_time = time.perf_counter() - t0
    layer_label = f"Layer {layer}" if layer == 2 else "Layer 3"
    print(f"Evolution ({layer_label}): {len(results)} unique alphas in {evolve_time:.2f}s")
    print(f"Archive: {archive.size}/{archive.capacity} cells filled ({archive.coverage:.1%} coverage)")
    print(f"Total time: {total_time:.2f}s\n")

    top = archive.best(args.top)
    print(f"Top {min(args.top, len(top))} alphas by fitness:")
    print(f"{'Rank':>4}  {'Fitness':>8}  Expression")
    print("-" * 70)
    for i, (expr, fit) in enumerate(top):
        print(f"{i + 1:>4}  {fit:>8.3f}  {to_string(expr)}")


def cmd_validate(args: argparse.Namespace) -> None:
    _warn_deprecated_live(args)
    cfg = _load_config(args.config)
    features = _make_features(args.asset)

    # Parse expression
    expr = parse(args.expr)
    print(f"Alpha: {to_string(expr)}")

    if args.synthetic:
        n_days = args.days
        data = _synthetic_data(features, n_days, seed=args.seed + 1000)
    else:
        data, n_days = _real_data(features, cfg, eval_window=args.eval_window)

    price_feat = features[0]
    prices = data[price_feat]

    # Evaluate
    sig = expr.evaluate(data)
    if isinstance(sig, (int, float, np.floating)):
        sig = np.full(n_days, float(sig))

    # Purged Walk-Forward CV
    engine = BacktestEngine(
        CostModel(cfg.backtest.commission_pct, cfg.backtest.slippage_pct)
    )
    cv = purged_walk_forward(
        sig, prices, engine,
        n_folds=cfg.validation.n_cv_folds,
        embargo=cfg.validation.embargo_days,
    )

    print(f"\nPurged Walk-Forward CV ({cv.n_folds} folds):")
    print(f"  OOS Sharpe:     {cv.oos_sharpe:>8.3f} +/- {cv.oos_sharpe_std:.3f}")
    print(f"  OOS Return:     {cv.oos_return:>7.1%}")
    print(f"  OOS Max DD:     {cv.oos_max_dd:>7.1%}")
    print(f"  Fold Sharpes:   {[f'{s:.3f}' for s in cv.fold_sharpes]}")

    passed = cv.oos_sharpe >= cfg.validation.oos_sharpe_min
    print(f"\n  Gate (OOS Sharpe >= {cfg.validation.oos_sharpe_min}): {'PASS' if passed else 'FAIL'}")


def cmd_forward(args: argparse.Namespace) -> None:
    from alpha_os.forward.runner import ForwardRunner, ForwardConfig
    from alpha_os.pipeline.scheduler import PipelineScheduler, SchedulerConfig

    cfg = _load_config(args.config)
    interval = args.interval or cfg.forward.check_interval
    fwd_cfg = ForwardConfig(
        check_interval=interval,
        min_forward_days=cfg.forward.min_forward_days,
        degradation_window=cfg.forward.degradation_window,
    )
    runner = ForwardRunner(asset=args.asset, config=cfg, forward_config=fwd_cfg)

    if args.summary:
        runner.print_summary()
        runner.close()
        return

    if args.once or not args.schedule:
        result = runner.run_cycle()
        runner.print_summary()
        runner.close()
        print(
            f"\nCycle: {result.n_evaluated} evaluated, {result.n_degraded} degraded, "
            f"{result.n_dormant} dormant, {result.n_revived} revived in {result.elapsed:.1f}s"
        )
        return

    def cycle():
        runner.run_cycle()

    scheduler = PipelineScheduler(
        run_fn=cycle,
        config=SchedulerConfig(interval_seconds=interval),
    )
    try:
        scheduler.start()
    finally:
        runner.close()


def _print_paper_result(result) -> None:
    print(f"\n{'='*60}")
    print(f"Paper Trading Cycle: {result.date}")
    print(f"{'='*60}")
    print(f"  Signal:     {result.combined_signal:+.4f}")
    print(f"  Risk Scale: DD={result.dd_scale:.2f} Vol={result.vol_scale:.2f}")
    print(f"  Trades:     {len(result.fills)}")
    for f in result.fills:
        print(f"    {f.side.upper():>4} {f.qty:.6f} {f.symbol} @ ${f.price:,.2f}")
    print(f"  Portfolio:  ${result.portfolio_value:,.2f}")
    print(f"  Daily P&L:  ${result.daily_pnl:+,.2f} ({result.daily_return:+.2%})")
    print(f"  Alphas:     {result.n_alphas_active} active, {result.n_alphas_evaluated} evaluated")


def cmd_paper(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    interval = args.interval or cfg.forward.check_interval

    if args.backfill:
        _cmd_paper_backfill(args, cfg)
        return

    from alpha_os.paper.trader import PaperTrader
    from alpha_os.pipeline.scheduler import PipelineScheduler, SchedulerConfig

    trader = PaperTrader(asset=args.asset, config=cfg)

    if args.summary:
        trader.print_status()
        trader.close()
        return

    if args.once or not args.schedule:
        result = trader.run_cycle()
        _print_paper_result(result)
        trader.print_status()
        trader.close()
        return

    def cycle():
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


def _cmd_paper_backfill(args: argparse.Namespace, cfg) -> None:
    from alpha_os.paper.simulator import run_backfill

    if not args.start or not args.end:
        print("Error: --backfill requires --start and --end dates")
        sys.exit(1)

    print(f"Running backfill simulation: {args.start} to {args.end} ({args.asset})")
    result = run_backfill(
        asset=args.asset,
        config=cfg,
        start_date=args.start,
        end_date=args.end,
    )

    print(f"\nBackfill Simulation: {args.start} to {args.end}")
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


def _build_pipeline_config(
    config: Config, pop_size: int, generations: int,
):
    """Build PipelineConfig from global Config + CLI args."""
    from alpha_os.pipeline.runner import PipelineConfig
    from alpha_os.governance.gates import GateConfig

    return PipelineConfig(
        gp=GPConfig(
            pop_size=pop_size,
            n_generations=generations,
            max_depth=config.generation.max_depth,
        ),
        gate=GateConfig(
            oos_sharpe_min=config.validation.oos_sharpe_min,
            pbo_max=config.validation.pbo_max,
            dsr_pvalue_max=config.validation.dsr_pvalue_max,
            min_n_days=config.backtest.min_days,
        ),
        commission_pct=config.backtest.commission_pct,
        slippage_pct=config.backtest.slippage_pct,
        n_cv_folds=config.validation.n_cv_folds,
        embargo_days=config.validation.embargo_days,
        eval_window_days=config.backtest.eval_window_days,
    )


def _run_evolution(trader, config: Config, pipeline_config) -> None:
    """Run alpha evolution using trader's data and registry."""
    from alpha_os.pipeline.runner import PipelineRunner

    logger = logging.getLogger(__name__)

    # Sync data before evolution
    try:
        trader.store.sync(trader.features)
    except Exception:
        logger.warning("API sync failed before evolution — using cached data")

    matrix = trader.store.get_matrix(trader.features)
    if len(matrix) < config.backtest.min_days:
        logger.warning(
            "Insufficient data for evolution (%d rows, need %d)",
            len(matrix), config.backtest.min_days,
        )
        return

    matrix = matrix.bfill().fillna(0)
    data = {col: matrix[col].values for col in matrix.columns}
    prices = data[trader.price_signal]

    runner = PipelineRunner(
        features=trader.features,
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


def _print_validation_report(report) -> None:
    print(f"\n--- Testnet Validation Report ({report.date}) ---")
    print(f"  Cycle OK:       {report.cycle_completed}")
    print(f"  Recon match:    {report.reconciliation_match}")
    print(f"  CB halted:      {report.circuit_breaker_halted}")
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
):
    """Create trader, circuit breaker, and validator for one asset."""
    from alpha_os.paper.trader import Trader
    from alpha_os.risk.circuit_breaker import CircuitBreaker

    adir = asset_data_dir(asset)
    cb = CircuitBreaker.load(path=adir / "metrics" / "circuit_breaker.json")
    cfg.trading.initial_capital = capital

    executor = None
    if is_crypto(asset):
        from alpha_os.execution.binance import BinanceExecutor
        signal_name = price_signal(asset)
        market_symbol = f"{asset}/USDT"
        symbol_map = {signal_name: market_symbol}
        executor = BinanceExecutor(testnet=testnet, symbol_map=symbol_map)

    trader = Trader(asset=asset, config=cfg, executor=executor, circuit_breaker=cb)

    validator = None
    if testnet and is_crypto(asset):
        from alpha_os.validation.testnet import TestnetValidator
        validator = TestnetValidator(
            state_path=adir / "metrics" / "testnet_validation.json",
            report_path=adir / "metrics" / "testnet_reports.jsonl",
            target_days=cfg.testnet.target_success_days,
            max_slippage_bps=cfg.testnet.max_acceptable_slippage_bps,
        )

    return trader, cb, validator


def cmd_live(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    interval = args.interval or cfg.forward.check_interval
    testnet = not args.real
    asset_list = _resolve_asset_list(args)

    # File logging
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"live_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )

    from alpha_os.pipeline.scheduler import PipelineScheduler, SchedulerConfig

    if args.real:
        print("=" * 60)
        print("WARNING: REAL TRADING MODE — real money on Binance.")
        print("Press Ctrl+C within 5 seconds to abort...")
        print("=" * 60)
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    mode = "TESTNET" if testnet else "REAL"
    print(f"Live trading [{mode}]: assets={','.join(asset_list)}, interval={interval}s")

    # Initialize per-asset contexts
    contexts: dict[str, tuple] = {}
    for asset in asset_list:
        trader, cb, validator = _setup_asset_context(
            asset, cfg, testnet, args.capital,
        )
        contexts[asset] = (trader, cb, validator)
        print(f"  {asset}: {price_signal(asset)} → {asset}/USDT")

    if args.summary:
        for asset in asset_list:
            print(f"\n--- {asset} ---")
            contexts[asset][0].print_status()
        for trader, _, _ in contexts.values():
            trader.close()
        return

    from alpha_os.alpha.registry import AlphaState

    pipeline_cfg = _build_pipeline_config(cfg, args.pop_size, args.generations)

    def _needs_evolution(trader):
        active = trader.registry.list_by_state(AlphaState.ACTIVE)
        probation = trader.registry.list_by_state(AlphaState.PROBATION)
        return len(active) + len(probation) == 0

    def _run_asset_validation(result, recon, cb, validator):
        if validator is None:
            return
        report = validator.validate_cycle(
            result, recon, cb, result.fills,
            order_failures=getattr(result, "order_failures", 0),
        )
        _print_validation_report(report)
        validator.print_status()

    if args.once or not args.schedule:
        for asset in asset_list:
            print(f"\n{'='*40} {asset} {'='*40}")
            trader, cb, validator = contexts[asset]
            if _needs_evolution(trader):
                print(f"No alphas for {asset} — running evolution...")
                _run_evolution(trader, cfg, pipeline_cfg)
            result = trader.run_cycle()
            _print_paper_result(result)
            recon = trader.reconcile()
            _run_asset_validation(result, recon, cb, validator)
            trader.print_status()
        for trader, _, _ in contexts.values():
            trader.close()
        return

    last_evolve: dict[str, float] = {a: 0.0 for a in asset_list}
    logger = logging.getLogger(__name__)

    def cycle():
        for asset in asset_list:
            trader, cb, validator = contexts[asset]
            logger.info("--- %s cycle start ---", asset)
            now = time.time()
            evolve_interval = args.evolve_interval
            if evolve_interval > 0:
                if _needs_evolution(trader) or (now - last_evolve[asset]) >= evolve_interval:
                    logger.info("Running alpha evolution for %s...", asset)
                    _run_evolution(trader, cfg, pipeline_cfg)
                    last_evolve[asset] = now
            result = trader.run_cycle()
            _print_paper_result(result)
            recon = trader.reconcile()
            _run_asset_validation(result, recon, cb, validator)
            logger.info("--- %s cycle done ---", asset)

    scheduler = PipelineScheduler(
        run_fn=cycle,
        config=SchedulerConfig(interval_seconds=interval),
    )
    try:
        scheduler.start()
    finally:
        for trader, _, _ in contexts.values():
            trader.close()


def cmd_validate_testnet(args: argparse.Namespace) -> None:
    import json as _json

    from alpha_os.validation.testnet import TestnetValidator

    cfg = _load_config(getattr(args, "config", None))
    adir = asset_data_dir(args.asset)
    report_path = adir / "metrics" / "testnet_reports.jsonl"

    validator = TestnetValidator(
        state_path=adir / "metrics" / "testnet_validation.json",
        report_path=report_path,
        target_days=cfg.testnet.target_success_days,
        max_slippage_bps=cfg.testnet.max_acceptable_slippage_bps,
    )

    if args.reset:
        validator._state.consecutive_success_days = 0
        validator._state.passed = False
        validator._save_state()
        print("Reset consecutive success counter to 0.")
        return

    validator.print_status()

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
            r = _json.loads(line)
            status = "OK" if not r["has_errors"] else "ERROR"
            print(
                f"  {r['date']} [{status}] PV=${r['portfolio_value']:,.2f} "
                f"PnL=${r['daily_pnl']:+,.2f} fills={r['n_fills']}"
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
    elif args.command == "forward":
        cmd_forward(args)
    elif args.command == "paper":
        cmd_paper(args)
    elif args.command == "live":
        cmd_live(args)
    elif args.command == "validate-testnet":
        cmd_validate_testnet(args)
