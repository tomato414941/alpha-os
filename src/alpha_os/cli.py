"""CLI for alpha-os: generate, backtest, validate alpha factors."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.config import Config
from alpha_os.data.universe import price_signal, MACRO_SIGNALS
from alpha_os.dsl import parse, to_string
from alpha_os.dsl.generator import AlphaGenerator
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
    bt.add_argument("--days", type=int, default=500)
    bt.add_argument("--seed", type=int, default=42)
    bt.add_argument("--config", type=str, default=None)

    # validate
    val = sub.add_parser("validate", help="Validate an alpha with purged WF CV")
    val.add_argument("--expr", type=str, required=True)
    val.add_argument("--asset", type=str, default="NVDA")
    val.add_argument("--days", type=int, default=500)
    val.add_argument("--seed", type=int, default=42)
    val.add_argument("--config", type=str, default=None)

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
    return [price] + MACRO_SIGNALS


def _synthetic_data(features: list[str], n_days: int, seed: int) -> dict[str, np.ndarray]:
    """Generate synthetic price/signal data for offline testing."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for i, feat in enumerate(features):
        # Each feature: random walk with slight drift
        drift = rng.uniform(-0.0005, 0.001)
        vol = rng.uniform(0.005, 0.03)
        returns = rng.normal(drift, vol, n_days)
        prices = 100.0 * np.cumprod(1.0 + returns)
        data[feat] = prices
    return data


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
    cfg = _load_config(args.config)
    features = _make_features(args.asset)
    gen = AlphaGenerator(features=features, seed=args.seed)

    # Generate alphas
    t0 = time.perf_counter()
    alphas = gen.generate_random(args.count, max_depth=3)
    templates = gen.generate_from_templates()
    all_alphas = alphas + templates
    gen_time = time.perf_counter() - t0

    # Synthetic data
    data = _synthetic_data(features, args.days, seed=args.seed + 1000)
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
                sig = np.full(args.days, float(sig))
            if len(sig) != args.days:
                continue
            if np.all(np.isnan(sig)):
                continue
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

    # Rank by Sharpe
    ranked = sorted(
        zip(valid_alphas, results),
        key=lambda x: x[1].sharpe,
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


def cmd_validate(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    features = _make_features(args.asset)

    # Parse expression
    expr = parse(args.expr)
    print(f"Alpha: {to_string(expr)}")

    # Synthetic data
    data = _synthetic_data(features, args.days, seed=args.seed + 1000)
    price_feat = features[0]
    prices = data[price_feat]

    # Evaluate
    sig = expr.evaluate(data)
    if isinstance(sig, (int, float, np.floating)):
        sig = np.full(args.days, float(sig))

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
    elif args.command == "validate":
        cmd_validate(args)
