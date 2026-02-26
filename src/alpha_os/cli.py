"""CLI for alpha-os: generate, backtest, validate alpha factors."""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np

from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.config import Config, DATA_DIR
from alpha_os.data.universe import price_signal, load_daily_signals, SIGNAL_NOISE_DB
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
    fwd.add_argument("--asset", type=str, default="NVDA")
    fwd.add_argument("--config", type=str, default=None)

    # paper
    ppr = sub.add_parser("paper", help="Paper trade with adopted alphas")
    ppr.add_argument("--once", action="store_true", help="Run one cycle and exit")
    ppr.add_argument("--schedule", action="store_true", help="Run on interval")
    ppr.add_argument("--summary", action="store_true", help="Print summary and exit")
    ppr.add_argument("--backfill", action="store_true",
                     help="Run historical simulation over date range")
    ppr.add_argument("--start", type=str, default=None,
                     help="Start date for backfill (ISO format, e.g. 2025-06-01)")
    ppr.add_argument("--end", type=str, default=None,
                     help="End date for backfill (ISO format, e.g. 2026-02-25)")
    ppr.add_argument("--asset", type=str, default="BTC")
    ppr.add_argument("--config", type=str, default=None)

    return parser


def _load_config(config_path: str | None) -> Config:
    from pathlib import Path
    if config_path:
        return Config.load(Path(config_path))
    return Config.load()


def _make_features(asset: str) -> list[str]:
    """Feature names available for alpha generation."""
    try:
        price = price_signal(asset)
    except KeyError:
        price = asset.lower()
    daily = load_daily_signals()
    # price signal first, then all daily signals (deduplicated)
    seen = {price}
    result = [price]
    for s in daily:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


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
) -> tuple[dict[str, np.ndarray], int]:
    """Load real data: cache-first, import from signal-noise, sync from API."""
    from alpha_os.data.store import DataStore

    db_path = DATA_DIR / "alpha_cache.db"

    store = DataStore(db_path)

    # Import from signal-noise DB (fast, local)
    if SIGNAL_NOISE_DB.exists():
        store.import_from_signal_noise(SIGNAL_NOISE_DB, features)

    # Try API sync (best-effort, for signals not in signal-noise)
    client = None
    try:
        from alpha_os.data.client import SignalClient
        client = SignalClient(
            base_url=config.api.base_url,
            timeout=config.api.timeout,
        )
        if client.health():
            print(f"Syncing from {config.api.base_url} ...")
            store.sync(features)
        else:
            print(f"API unavailable at {config.api.base_url} — using cache")
    except Exception:
        pass

    matrix = store.get_matrix(features)

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
        key=lambda x: x[1].sharpe if np.isfinite(x[1].sharpe) else -999.0,
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
    features = _make_features(args.asset)

    if args.synthetic:
        n_days = args.days
        data = _synthetic_data(features, n_days, seed=args.seed + 1000)
    else:
        data, n_days = _real_data(features, cfg, eval_window=args.eval_window)

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
                return -999.0
            if not np.all(np.isfinite(sig)):
                sig = np.where(np.isfinite(sig), sig, 0.0)
            result = engine.run(sig, prices)
            return result.sharpe if np.isfinite(result.sharpe) else -999.0
        except Exception:
            return -999.0

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
    print(f"Evolution: {len(results)} unique alphas in {evolve_time:.2f}s")
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
    fwd_cfg = ForwardConfig(
        check_interval=cfg.forward.check_interval,
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
        config=SchedulerConfig(interval_seconds=fwd_cfg.check_interval),
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
        config=SchedulerConfig(interval_seconds=cfg.forward.check_interval),
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
