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
from alpha_os.config import (
    Config,
    DATA_DIR,
    HYPOTHESIS_OBSERVATIONS_DB_NAME,
    HYPOTHESES_DB,
    SIGNAL_CACHE_DB,
    SIGNAL_CACHE_L2_DB,
    asset_data_dir,
)
from alpha_os.runtime_lock import RuntimeLockBusy, hold_runtime_lock, runtime_lock_path
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.universe import is_crypto, is_equity, infer_venue, price_signal, build_feature_list, build_hourly_feature_list
from alpha_os.dsl import parse, to_string
from alpha_os.runtime_profile import build_runtime_profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-os",
        description="Agentic Alpha OS — hypotheses-first runtime and research CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # generate (legacy top-level alias; prefer `research generate`)
    gen = sub.add_parser("generate", help=argparse.SUPPRESS)
    gen.add_argument("--count", type=int, default=5000)
    gen.add_argument("--asset", type=str, default="NVDA")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--config", type=str, default=None)

    # backtest (legacy top-level alias; prefer `research backtest`)
    bt = sub.add_parser("backtest", help=argparse.SUPPRESS)
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

    # evolve (legacy top-level alias; prefer `research evolve`)
    evo = sub.add_parser("evolve", help=argparse.SUPPRESS)
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

    # validate (legacy top-level alias; prefer `research validate`)
    val = sub.add_parser("validate", help=argparse.SUPPRESS)
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

    # evaluate (legacy top-level alias; prefer `research evaluate`)
    eva = sub.add_parser("evaluate", help=argparse.SUPPRESS)
    eva.add_argument("--expr", type=str, required=True, help="DSL expression string")
    eva.add_argument("--config", type=str, default=None)

    # produce-predictions
    pp = sub.add_parser("produce-predictions", help="Evaluate active hypotheses and write to prediction store")
    pp.add_argument("--asset", type=str, default="BTC")
    pp.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when no hypothesis predictions are written",
    )
    pp.add_argument("--config", type=str, default=None)

    # produce-classical (legacy top-level alias; prefer `research produce-classical`)
    pc = sub.add_parser("produce-classical", help=argparse.SUPPRESS)
    pc.add_argument("--config", type=str, default=None)

    # paper
    ppr = sub.add_parser("paper", help="Paper trade with live hypotheses")
    ppr.add_argument("--once", action="store_true", help="Run one cycle and exit")
    ppr.add_argument("--schedule", action="store_true", help=argparse.SUPPRESS)
    ppr.add_argument("--summary", action="store_true", help=argparse.SUPPRESS)
    ppr.add_argument("--interval", type=int, default=None,
                     help="Override check_interval in seconds (default: from config)")
    ppr.add_argument("--replay", action="store_true", help=argparse.SUPPRESS)
    ppr.add_argument("--start", type=str, default=None, help=argparse.SUPPRESS)
    ppr.add_argument("--end", type=str, default=None, help=argparse.SUPPRESS)
    ppr.add_argument("--sizing-mode", type=str, default="runtime",
                     choices=["runtime", "raw_mean", "compare"],
                     help=argparse.SUPPRESS)
    ppr.add_argument("--asset", type=str, default="BTC")
    ppr.add_argument("--config", type=str, default=None)

    # trade
    trd = sub.add_parser("trade", help="Trade on Binance (testnet by default)")
    trd.add_argument("--once", action="store_true", help="Run one cycle and exit")
    trd.add_argument("--schedule", action="store_true", help=argparse.SUPPRESS)
    trd.add_argument("--summary", action="store_true", help=argparse.SUPPRESS)
    trd.add_argument("--interval", type=int, default=None,
                     help="Override check_interval in seconds (default: from config)")
    trd.add_argument("--real", action="store_true",
                     help="Use real Binance (default is testnet)")
    trd.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive confirmation prompts for unattended runs",
    )
    trd.add_argument("--capital", type=float, default=10000.0,
                     help="Initial capital for tracking (default: 10000)")
    trd.add_argument("--asset", type=str, default="BTC")
    trd.add_argument("--assets", type=str, default=None,
                     help="Comma-separated asset list (e.g. BTC,ETH,SOL)")
    trd.add_argument("--config", type=str, default=None)
    trd.add_argument("--evolve-interval", type=int, default=86400, help=argparse.SUPPRESS)
    trd.add_argument("--pop-size", type=int, default=200, help=argparse.SUPPRESS)
    trd.add_argument("--generations", type=int, default=30, help=argparse.SUPPRESS)
    trd.add_argument("--event-driven", action="store_true", help=argparse.SUPPRESS)
    trd.add_argument("--debounce", type=int, default=None, help=argparse.SUPPRESS)
    trd.add_argument("--venue", type=str, default=None,
                     choices=["binance", "alpaca", "polymarket", "paper"],
                     help="Trading venue (default: auto-detect from asset)")
    trd.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on overlapping runs or empty/failed oneshot trade cycles",
    )

    hseed = sub.add_parser(
        "hypothesis-seeder",
        help="Register random DSL and fixed seed hypotheses into the hypotheses store",
    )
    hseed.add_argument("--config", type=str, default=None)

    seh = sub.add_parser(
        "score-exploratory-hypotheses",
        help="Compute research-quality metadata for exploratory random DSL hypotheses",
    )
    seh.add_argument("--asset", type=str, default="BTC")
    seh.add_argument("--config", type=str, default=None)
    seh.add_argument("--limit", type=int, default=None)
    seh.add_argument("--dry-run", action="store_true")

    ugen = sub.add_parser(
        "unified-generator",
        help=argparse.SUPPRESS,
    )
    ugen.add_argument("--config", type=str, default=None)

    pg = sub.add_parser(
        "enqueue-discovery-pool",
        help=argparse.SUPPRESS,
    )
    pg.add_argument("--asset", type=str, default="BTC")
    pg.add_argument("--config", type=str, default=None)
    pg.add_argument("--limit", type=int, default=None)
    pg.add_argument("--dry-run", action="store_true")

    # admission-daemon (Pipeline v2)
    adm_d = sub.add_parser("admission-daemon", help=argparse.SUPPRESS)
    adm_d.add_argument("--asset", type=str, default="BTC")
    adm_d.add_argument("--config", type=str, default=None)

    psc = sub.add_parser(
        "prune-stale-candidates",
        help=argparse.SUPPRESS,
    )
    psc.add_argument("--asset", type=str, default="BTC")
    psc.add_argument("--max-age-days", type=int, default=7)
    psc.add_argument("--dry-run", action="store_true")

    # lifecycle (legacy compat alias)
    lc_d = sub.add_parser("lifecycle", help=argparse.SUPPRESS)
    lc_d.add_argument("--asset", type=str, default="BTC")
    lc_d.add_argument("--config", type=str, default=None)

    rat = sub.add_parser(
        "rebalance-allocation-trust",
        help="Rebase active hypothesis stake values onto the current allocation-trust model",
    )
    rat.add_argument("--asset", type=str, default="BTC")
    rat.add_argument("--config", type=str, default=None)
    rat.add_argument("--dry-run", action="store_true")

    alb = sub.add_parser(
        "analyze-live-breadth",
        help="Analyze historical signal breadth for capital-backed bootstrap hypotheses",
    )
    alb.add_argument("--asset", type=str, default="BTC")
    alb.add_argument("--config", type=str, default=None)
    alb.add_argument("--lookback", type=int, default=252)
    alb.add_argument("--top-pairs", type=int, default=5)

    alc = sub.add_parser(
        "analyze-latest-combine",
        help="Decompose the latest selected alpha signals by runtime cohort",
    )
    alc.add_argument("--asset", type=str, default="BTC")
    alc.add_argument("--config", type=str, default=None)
    alc.add_argument("--top", type=int, default=5)

    abr = sub.add_parser(
        "analyze-batch-research",
        help="Summarize batch-research exploratory hypotheses and why they are or are not capital-backed",
    )
    abr.add_argument("--asset", type=str, default="BTC")
    abr.add_argument("--config", type=str, default=None)
    abr.add_argument("--top", type=int, default=5)
    abr.add_argument(
        "--families",
        type=str,
        default=None,
        help="Comma-separated feature families to focus on",
    )

    att = sub.add_parser(
        "analyze-trade-transition",
        help="Run one paper trade cycle and summarize pre/post batch research transitions",
    )
    att.add_argument("--asset", type=str, default="BTC")
    att.add_argument("--config", type=str, default=None)
    att.add_argument("--top", type=int, default=5)
    att.add_argument(
        "--families",
        type=str,
        default=None,
        help="Comma-separated feature families to focus on",
    )

    bor = sub.add_parser(
        "backfill-observation-returns",
        help="Backfill observation-only forward returns for active hypotheses from cached history",
    )
    bor.add_argument("--asset", type=str, default="BTC")
    bor.add_argument("--config", type=str, default=None)
    bor.add_argument("--days", type=int, default=30)
    bor.add_argument(
        "--apply-lifecycle",
        action="store_true",
        help="Recompute allocation trust after backfilling observation returns",
    )

    # replay-experiment
    rex = sub.add_parser(
        "replay-experiment",
        help=argparse.SUPPRESS,
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
        help=argparse.SUPPRESS,
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

    ssc = sub.add_parser(
        "sync-signal-cache",
        help="Sync a bounded set of signals into the local signal cache",
    )
    ssc.add_argument("--asset", type=str, default="BTC")
    ssc.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Comma-separated asset list (default: use --asset only)",
    )
    ssc.add_argument(
        "--signals",
        type=str,
        default=None,
        help="Comma-separated explicit signal names to sync",
    )
    ssc.add_argument(
        "--from-hypotheses",
        action="store_true",
        help="Sync signals required by active hypotheses for the selected assets",
    )
    ssc.add_argument(
        "--resolution",
        type=str,
        default="1d",
        help="Signal resolution (default: 1d)",
    )
    ssc.add_argument(
        "--min-history-days",
        type=int,
        default=0,
        help="Require at least this many cached rows per signal",
    )
    ssc.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when sync cannot populate every requested signal",
    )
    ssc.add_argument("--config", type=str, default=None)

    afl = sub.add_parser(
        "alpha-funnel",
        help=argparse.SUPPRESS,
    )
    afl.add_argument("--asset", type=str, default="BTC")

    legacy = sub.add_parser(
        "legacy",
        help="Run archived registry and migration commands",
    )
    legacy_sub = legacy.add_subparsers(dest="legacy_command", required=True)

    lgen = legacy_sub.add_parser(
        "unified-generator",
        help="Run the legacy unified generator flow",
    )
    lgen.add_argument("--config", type=str, default=None)

    lpg = legacy_sub.add_parser(
        "enqueue-discovery-pool",
        help="Enqueue top discovery-pool entries into the admission queue",
    )
    lpg.add_argument("--asset", type=str, default="BTC")
    lpg.add_argument("--config", type=str, default=None)
    lpg.add_argument("--limit", type=int, default=None)
    lpg.add_argument("--dry-run", action="store_true")

    ladm = legacy_sub.add_parser(
        "admission-daemon",
        help="Run the legacy candidate admission daemon",
    )
    ladm.add_argument("--asset", type=str, default="BTC")
    ladm.add_argument("--config", type=str, default=None)

    lpsc = legacy_sub.add_parser(
        "prune-stale-candidates",
        help="Reject stale pending candidates outside the active discovery/manual sources",
    )
    lpsc.add_argument("--asset", type=str, default="BTC")
    lpsc.add_argument("--max-age-days", type=int, default=7)
    lpsc.add_argument("--dry-run", action="store_true")

    llc = legacy_sub.add_parser(
        "lifecycle",
        help="Run the legacy daily stake-update daemon",
    )
    llc.add_argument("--asset", type=str, default="BTC")
    llc.add_argument("--config", type=str, default=None)

    lafl = legacy_sub.add_parser(
        "alpha-funnel",
        help="Show discovery-pool to deployed-alpha funnel counts",
    )
    lafl.add_argument("--asset", type=str, default="BTC")

    research = sub.add_parser(
        "research",
        help="Run bounded research and replay commands",
    )
    research_sub = research.add_subparsers(dest="research_command", required=True)

    rgen = research_sub.add_parser("generate", help="Generate alpha expressions")
    rgen.add_argument("--count", type=int, default=5000)
    rgen.add_argument("--asset", type=str, default="NVDA")
    rgen.add_argument("--seed", type=int, default=42)
    rgen.add_argument("--config", type=str, default=None)

    rbt = research_sub.add_parser("backtest", help="Backtest generated alphas")
    rbt.add_argument("--count", type=int, default=5000)
    rbt.add_argument("--top", type=int, default=20)
    rbt.add_argument("--asset", type=str, default="NVDA")
    rbt.add_argument("--days", type=int, default=500, help="Number of days (--synthetic only)")
    rbt.add_argument("--seed", type=int, default=42)
    rbt.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random-walk data instead of real data",
    )
    rbt.add_argument(
        "--eval-window",
        type=int,
        default=0,
        help="Evaluation window in days (0=all data, e.g. 200 for recent)",
    )
    rbt.add_argument("--config", type=str, default=None)

    revo = research_sub.add_parser("evolve", help="Evolve alphas via GP + MAP-Elites")
    revo.add_argument("--pop-size", type=int, default=200)
    revo.add_argument("--generations", type=int, default=30)
    revo.add_argument("--asset", type=str, default="NVDA")
    revo.add_argument("--days", type=int, default=500, help="Number of days (--synthetic only)")
    revo.add_argument("--top", type=int, default=20)
    revo.add_argument("--seed", type=int, default=42)
    revo.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random-walk data instead of real data",
    )
    revo.add_argument(
        "--eval-window",
        type=int,
        default=0,
        help="Evaluation window in days (0=all data, e.g. 200 for recent)",
    )
    revo.add_argument(
        "--layer",
        type=int,
        default=3,
        choices=[2, 3],
        help="Alpha layer: 2=hourly tactical, 3=daily strategic (default)",
    )
    revo.add_argument("--config", type=str, default=None)

    rval = research_sub.add_parser("validate", help="Validate an alpha with purged WF CV")
    rval.add_argument("--expr", type=str, required=True)
    rval.add_argument("--asset", type=str, default="NVDA")
    rval.add_argument("--days", type=int, default=500, help="Number of days (--synthetic only)")
    rval.add_argument("--seed", type=int, default=42)
    rval.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random-walk data instead of real data",
    )
    rval.add_argument(
        "--eval-window",
        type=int,
        default=0,
        help="Evaluation window in days (0=all data, e.g. 200 for recent)",
    )
    rval.add_argument(
        "--layer",
        type=int,
        default=3,
        choices=[2, 3],
        help="Alpha layer: 2=hourly tactical, 3=daily strategic (default)",
    )
    rval.add_argument("--config", type=str, default=None)

    reva = research_sub.add_parser("evaluate", help="Evaluate expression with multi-horizon IC")
    reva.add_argument("--expr", type=str, required=True, help="DSL expression string")
    reva.add_argument("--config", type=str, default=None)

    rpc = research_sub.add_parser(
        "produce-classical",
        help="Compute classical indicators and write to prediction store",
    )
    rpc.add_argument("--config", type=str, default=None)

    rpr = research_sub.add_parser(
        "paper-replay",
        help="Run the legacy historical paper replay path",
    )
    rpr.add_argument("--start", type=str, required=True, help="Start date for replay (ISO format)")
    rpr.add_argument("--end", type=str, required=True, help="End date for replay (ISO format)")
    rpr.add_argument(
        "--sizing-mode",
        type=str,
        default="runtime",
        choices=["runtime", "raw_mean", "compare"],
        help="Legacy replay sizing mode",
    )
    rpr.add_argument("--asset", type=str, default="BTC")
    rpr.add_argument("--config", type=str, default=None)

    rrex = research_sub.add_parser(
        "replay-experiment",
        help="Run a legacy replay experiment and persist the artifact",
    )
    rrex.add_argument("--name", required=True, help="Experiment name")
    rrex.add_argument("--asset", type=str, default="BTC")
    rrex.add_argument("--config", type=str, default=None)
    rrex.add_argument("--start", required=True, help="Replay start date (YYYY-MM-DD)")
    rrex.add_argument("--end", required=True, help="Replay end date (YYYY-MM-DD)")
    rrex.add_argument(
        "--managed-alpha-mode",
        choices=["current", "admission"],
        default="current",
        help="Use the current managed-alpha set as-is or rebuild it from admission rules first",
    )
    rrex.add_argument(
        "--source",
        choices=["alphas", "candidates"],
        default="candidates",
        help="Admission replay source when --managed-alpha-mode=admission",
    )
    rrex.add_argument(
        "--fail-state",
        choices=["rejected", "dormant"],
        default="rejected",
        help="Fallback state for records that fail admission replay",
    )
    rrex.add_argument(
        "--deployment-mode",
        choices=["current", "refresh"],
        default="current",
        help="Use the current deployed alpha set or refresh it inside the experiment",
    )
    rrex.add_argument(
        "--sizing-mode",
        type=str,
        default="runtime",
        choices=["runtime", "raw_mean"],
        help="Replay sizing mode",
    )
    rrex.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override merged config via dotted path, e.g. lifecycle.candidate_quality_min=1.10",
    )
    rrex.add_argument("--notes", default="", help="Optional experiment notes")

    rrmx = research_sub.add_parser(
        "replay-matrix",
        help="Run a TOML-defined replay experiment matrix",
    )
    rrmx.add_argument("--manifest", required=True, help="Path to TOML matrix manifest")
    rrmx.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers for historical replay runs",
    )

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

    db_path = SIGNAL_CACHE_L2_DB if resolution != "1d" else SIGNAL_CACHE_DB

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
    from alpha_os.dsl.evaluator import FAILED_FITNESS

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
    from alpha_os.dsl.evaluator import FAILED_FITNESS, sanitize_signal
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
    print(f"  Active:     {result.n_active_hypotheses} hypotheses")
    print(f"  Live:       {result.n_live_hypotheses} hypotheses")
    print(f"  Shortlist:  {result.n_shortlist_candidates} candidates")
    print(f"  Selected:   {result.n_selected_hypotheses} hypotheses")
    print(f"  Signals:    {result.n_signals_evaluated} evaluated")
    if result.n_skipped_deadband > 0:
        print(f"  Skips:      deadband={result.n_skipped_deadband}")
    if getattr(result, "n_skipped_no_delta", 0) > 0:
        print(f"  Skips:      no_delta={result.n_skipped_no_delta}")
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

    trader = Trader(asset=args.asset, config=cfg)

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


def _cmd_paper_replay(args: argparse.Namespace, cfg) -> None:
    from alpha_os.research.replay_simulator import run_replay

    if not args.start or not args.end:
        print("Error: --replay requires --start and --end dates")
        sys.exit(1)

    print(f"Running legacy historical replay: {args.start} to {args.end} ({args.asset})")
    def _print_replay_result(label: str, result) -> None:
        print(f"\nLegacy Historical Replay [{label}]: {args.start} to {args.end}")
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


def cmd_paper_replay(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    _cmd_paper_replay(args, cfg)


def _build_pipeline_config(
    config: Config, pop_size: int, generations: int,
):
    """Build PipelineConfig from global Config + CLI args."""
    from alpha_os.evolution.gp import GPConfig
    from alpha_os.research.pipeline_runner import PipelineConfig

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
    from alpha_os.research.pipeline_runner import PipelineRunner

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


def _print_testnet_report(report) -> None:
    print(f"\n--- Testnet Readiness Report ({report.date}) ---")
    if getattr(report, "profile_id", ""):
        suffix = f" ({report.profile_commit[:8]})" if report.profile_commit else ""
        print(f"  Profile:        {report.profile_id[:12]}{suffix}")
    print(f"  Cycle OK:       {report.cycle_completed}")
    print(f"  Recon match:    {report.reconciliation_match}")
    print(f"  CB halted:      {report.circuit_breaker_halted}")
    print(f"  Active:         {report.n_active_hypotheses} hypotheses")
    print(f"  Live:           {report.n_live_hypotheses} hypotheses")
    print(f"  Shortlist:      {report.n_shortlist_candidates} candidates")
    print(f"  Selected:       {report.n_selected_hypotheses} hypotheses")
    print(f"  Signals:        {report.n_signals_evaluated} evaluated")
    if report.n_skipped_deadband > 0:
        print(f"  Skips:          deadband={report.n_skipped_deadband}")
    if getattr(report, "n_skipped_no_delta", 0) > 0:
        print(f"  Skips:          no_delta={report.n_skipped_no_delta}")
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
    if getattr(args, "non_interactive", False):
        print("Real trading mode: non-interactive confirmation bypass enabled.")
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
    registry = trader.registry
    asset = getattr(trader, "asset", None)

    if hasattr(registry, "list_by_state"):
        active = registry.list_by_state("active")
        return len(active) == 0
    if hasattr(registry, "list_active"):
        try:
            active = registry.list_active(asset=asset)
        except TypeError:
            active = registry.list_active()
        return len(active) == 0
    if hasattr(registry, "top_by_stake"):
        try:
            top = registry.top_by_stake(n=1, asset=asset)
        except TypeError:
            top = registry.top_by_stake(n=1)
        return len(top) == 0
    return True


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


def _build_live_returns_getter(forward_tracker, *, supports_short: bool):
    if forward_tracker is None:
        return None
    realizable_getter = getattr(forward_tracker, "get_hypothesis_realizable_returns", None)
    if realizable_getter is not None:
        def live_returns_for(hypothesis_id: str):
            return realizable_getter(hypothesis_id, supports_short=supports_short)

        return live_returns_for

    getter = getattr(forward_tracker, "get_hypothesis_returns", None)
    if getter is None:
        return None

    def live_returns_for(hypothesis_id: str):
        return getter(hypothesis_id)

    return live_returns_for


def _build_signal_activity_getter(portfolio_tracker, *, lookback: int, supports_short: bool):
    if portfolio_tracker is None:
        return None
    getter = getattr(portfolio_tracker, "get_hypothesis_signal_history", None)
    if getter is None:
        return None

    def signal_activity_for(hypothesis_id: str):
        values = [float(v) for v in getter(hypothesis_id, limit=lookback)]
        if not values:
            return 0.0, 0.0
        if supports_short:
            nonzero = sum(1 for v in values if abs(v) > 1e-12)
            mean_abs = sum(abs(v) for v in values) / len(values)
            return nonzero / len(values), mean_abs
        positive = [v for v in values if v > 1e-12]
        if not positive:
            return 0.0, 0.0
        mean_positive = sum(positive) / len(values)
        return len(positive) / len(values), mean_positive

    return signal_activity_for


def _run_hypothesis_lifecycle_update(trader, cfg: Config, result) -> dict[str, float]:
    from alpha_os.hypotheses import (
        apply_allocation_rebalance_plan,
        build_capped_allocation_rebalance_plan,
        record_daily_contributions,
    )

    signal_date = getattr(result, "date", "")
    if not signal_date:
        return {}
    hypothesis_signals = trader.portfolio_tracker.get_hypothesis_signals(signal_date)
    if not hypothesis_signals:
        return {}

    contribution_date = signal_date[:10]
    record_daily_contributions(
        trader.registry,
        asset=getattr(trader, "asset", "BTC"),
        date=contribution_date,
        predictions=hypothesis_signals,
        realized_return=getattr(result, "daily_return", 0.0),
    )
    forward_tracker = getattr(trader, "forward_tracker", None)
    live_returns_for = _build_live_returns_getter(
        forward_tracker,
        supports_short=cfg.trading.supports_short,
    )
    signal_activity_for = _build_signal_activity_getter(
        trader.portfolio_tracker,
        lookback=cfg.forward.degradation_window,
        supports_short=cfg.trading.supports_short,
    )
    asset = getattr(trader, "asset", "BTC")
    plan = build_capped_allocation_rebalance_plan(
        trader.registry,
        asset=asset,
        config=cfg,
        live_returns_for=live_returns_for,
        signal_activity_for=signal_activity_for,
    )
    updates = apply_allocation_rebalance_plan(trader.registry, plan)
    if updates:
        print(
            "Lifecycle summary: "
            f"date={contribution_date} updated={len(updates)} "
            f"rate={cfg.lifecycle.stake_update_rate:.2f}"
        )
    return updates


def _run_trade_once(
    *,
    asset_list: list[str],
    contexts: dict[str, tuple],
    cfg: Config,
    pipeline_cfg,
) -> list[tuple[str, object]]:
    use_lifecycle_daemon = cfg.lifecycle_daemon.enabled
    results: list[tuple[str, object]] = []
    for asset in asset_list:
        print(f"\n{'='*40} {asset} {'='*40}")
        trader, cb, readiness_checker = contexts[asset]
        if _needs_trade_evolution(trader):
            print(f"No alphas for {asset} — running evolution...")
            _run_evolution(trader, cfg, pipeline_cfg)
        result = trader.run_cycle(skip_lifecycle=use_lifecycle_daemon)
        _print_paper_result(result)
        recon = trader.reconcile()
        _run_trade_readiness_check(result, recon, cb, readiness_checker)
        if not use_lifecycle_daemon:
            _run_hypothesis_lifecycle_update(trader, cfg, result)
        trader.print_status()
        print(
            "Trade summary: "
            f"asset={asset} "
            f"status={_trade_once_status(result)} "
            f"live={result.n_live_hypotheses} "
            f"signals={result.n_signals_evaluated} "
            f"fills={len(result.fills)} "
            f"order_failures={result.order_failures}"
        )
        results.append((asset, result))
    return results


def _trade_once_strict_failure(asset: str, result) -> str | None:
    if getattr(result, "n_live_hypotheses", 0) <= 0:
        return f"{asset}: no live hypotheses"
    if getattr(result, "n_signals_evaluated", 0) <= 0:
        return f"{asset}: no signals evaluated"
    if getattr(result, "order_failures", 0) > 0:
        return f"{asset}: order failures={result.order_failures}"
    return None


def _trade_once_status(result) -> str:
    if getattr(result, "n_live_hypotheses", 0) <= 0:
        return "no_live_hypotheses"
    if getattr(result, "n_signals_evaluated", 0) <= 0:
        return "no_signals"
    if getattr(result, "order_failures", 0) > 0:
        return "order_failures"
    if getattr(result, "fills", None):
        return "traded"
    if getattr(result, "n_skipped_no_delta", 0) > 0:
        return "no_delta"
    if (
        getattr(result, "n_skipped_deadband", 0) > 0
        or getattr(result, "n_skipped_min_notional", 0) > 0
        or getattr(result, "n_skipped_rounded_to_zero", 0) > 0
    ):
        return "skipped"
    return "idle"


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
    last_lifecycle_date: dict[str, str] = {a: "" for a in asset_list}

    def cycle() -> None:
        nonlocal prev_raw_signals

        # Compute cross-asset neutralization from previous cycle
        signal_overrides: dict[str, float | None] = {a: None for a in asset_list}
        if len(asset_list) > 1 and len(prev_raw_signals) == len(asset_list):
            from alpha_os.hypotheses.combiner import cross_asset_neutralize
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
            if not use_lifecycle_daemon:
                contribution_date = getattr(result, "date", "")[:10]
                if contribution_date and contribution_date != last_lifecycle_date[asset]:
                    _run_hypothesis_lifecycle_update(trader, cfg, result)
                    last_lifecycle_date[asset] = contribution_date
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
        if getattr(args, "strict", False):
            sys.exit(1)
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
            results = _run_trade_once(
                asset_list=asset_list,
                contexts=contexts,
                cfg=cfg,
                pipeline_cfg=pipeline_cfg,
            )
            _close_trade_contexts(contexts)
            if getattr(args, "strict", False):
                failures = [
                    failure
                    for asset, result in results
                    if (failure := _trade_once_strict_failure(asset, result)) is not None
                ]
                if failures:
                    print("Strict mode failures:")
                    for failure in failures:
                        print(f"  - {failure}")
                    sys.exit(1)
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


def cmd_hypothesis_seeder(args: argparse.Namespace) -> None:
    """Run the hypothesis seeder daemon."""
    cfg = _load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    from alpha_os.daemon.hypothesis_seeder import HypothesisSeederDaemon

    daemon = HypothesisSeederDaemon(config=cfg)
    daemon.run()


def cmd_unified_generator(args: argparse.Namespace) -> None:
    """Backward-compatible alias for `hypothesis-seeder`."""
    cmd_hypothesis_seeder(args)


def cmd_enqueue_discovery_pool(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)

    from alpha_os.legacy.alpha_generator import enqueue_discovery_pool_candidates

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
    """Run the legacy daily stake-update daemon."""
    cfg = _load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    from alpha_os.legacy.lifecycle import LifecycleDaemon

    daemon = LifecycleDaemon(asset=args.asset, config=cfg)
    daemon.run()


def cmd_rebalance_allocation_trust(args: argparse.Namespace) -> None:
    from alpha_os.forward.tracker import HypothesisObservationTracker
    from alpha_os.data.store import DataStore
    from alpha_os.paper.tracker import PaperPortfolioTracker
    from alpha_os.hypotheses import (
        HypothesisStore,
        apply_allocation_rebalance_plan,
        build_capped_allocation_rebalance_plan,
    )

    cfg = _load_config(args.config)
    adir = asset_data_dir(args.asset)
    store = HypothesisStore(HYPOTHESES_DB)
    forward_tracker = HypothesisObservationTracker(adir / HYPOTHESIS_OBSERVATIONS_DB_NAME)
    portfolio_tracker = PaperPortfolioTracker(db_path=adir / "paper_trading.db")
    data_store = DataStore(SIGNAL_CACHE_DB)
    try:
        live_returns_for = _build_live_returns_getter(
            forward_tracker,
            supports_short=cfg.trading.supports_short,
        )
        signal_activity_for = _build_signal_activity_getter(
            portfolio_tracker,
            lookback=cfg.forward.degradation_window,
            supports_short=cfg.trading.supports_short,
        )
        plan = build_capped_allocation_rebalance_plan(
            store,
            asset=args.asset,
            config=cfg,
            live_returns_for=live_returns_for,
            signal_activity_for=signal_activity_for,
            data_store=data_store,
        )
        zeroed = sum(
            1 for entry in plan
            if entry.current_stake > 0 and entry.proposed_stake <= cfg.lifecycle.target_stake_floor
        )
        research_backed = sum(1 for entry in plan if entry.research_backed)
        live_proven = sum(1 for entry in plan if entry.live_proven)
        redundancy_capped = sum(1 for entry in plan if entry.redundancy_capped_by)
        research_candidate_capped = sum(1 for entry in plan if entry.research_candidate_capped)
        changed = sum(
            1 for entry in plan
            if abs(entry.proposed_stake - entry.current_stake) > 1e-12
        )
        mode = "DRY RUN" if args.dry_run else "APPLY"
        print(f"Allocation trust rebalance [{mode}]: asset={args.asset.upper()}")
        print(
            "  Summary: "
            f"active={len(plan)} changed={changed} zeroed={zeroed} "
            f"research_backed={research_backed} live_proven={live_proven} "
            f"research_candidate_capped={research_candidate_capped} "
            f"redundancy_capped={redundancy_capped}"
        )

        ranked = sorted(
            plan,
            key=lambda entry: abs(entry.proposed_stake - entry.current_stake),
            reverse=True,
        )
        for entry in ranked[:10]:
            print(
                "  "
                f"{entry.hypothesis_id}: {entry.current_stake:.3f} -> {entry.proposed_stake:.3f} "
                f"(target={entry.target_stake:.3f} boot={entry.bootstrap_trust_value:.3f} "
                f"conf={entry.confidence:.3f} n={entry.n_observations} "
                f"research_backed={entry.research_backed} live_proven={entry.live_proven}"
                f"{' research_candidate_capped=true' if entry.research_candidate_capped else ''}"
                f"{f' capped_by={entry.redundancy_capped_by} corr={entry.redundancy_correlation:.3f}' if entry.redundancy_capped_by else ''})"
            )

        if args.dry_run:
            return

        updates = apply_allocation_rebalance_plan(store, plan)
        print(
            "Rebalance summary: "
            f"updated={len(updates)} active={len(plan)} zeroed={zeroed}"
        )
    finally:
        portfolio_tracker.close()
        data_store.close()
        forward_tracker.close()
        store.close()


def cmd_analyze_live_breadth(args: argparse.Namespace) -> None:
    from alpha_os.data.store import DataStore
    from alpha_os.hypotheses.breadth import (
        analyze_capital_breadth,
        load_capital_backed_records,
        load_breadth_matrix,
    )
    from alpha_os.hypotheses.store import HypothesisStore

    _load_config(args.config)
    store = HypothesisStore(HYPOTHESES_DB)
    data_store = DataStore(SIGNAL_CACHE_DB)
    try:
        records = load_capital_backed_records(store, asset=args.asset)
        data = load_breadth_matrix(
            data_store,
            records,
            asset=args.asset,
            lookback=args.lookback,
        )
        report = analyze_capital_breadth(
            records,
            data=data,
            asset=args.asset,
            lookback=args.lookback,
            top_pairs=args.top_pairs,
        )
    finally:
        data_store.close()
        store.close()

    print(f"Live breadth ({args.asset.upper()})")
    print(
        "  Summary: "
        f"records={report.n_records} analyzed={report.n_analyzed} skipped={report.n_skipped} "
        f"lookback={report.lookback}"
    )
    print(
        "  Breadth: "
        f"mean_abs_corr={report.mean_abs_correlation:.3f} "
        f"effective={report.effective_breadth:.2f}"
    )
    if report.top_pairs:
        print("  TopPairs:")
        for pair in report.top_pairs:
            print(
                "    "
                f"{pair.left_id} <-> {pair.right_id}: "
                f"corr={pair.correlation:+.3f} "
                f"|corr|={pair.abs_correlation:.3f}"
            )
    if report.skipped_ids:
        skipped = ", ".join(report.skipped_ids[:10])
        suffix = " ..." if len(report.skipped_ids) > 10 else ""
        print(f"  Skipped:   {skipped}{suffix}")


def cmd_analyze_batch_research(args: argparse.Namespace) -> None:
    from alpha_os.hypotheses.batch_research_diagnostics import (
        build_batch_research_summary,
    )
    from alpha_os.hypotheses.store import HypothesisStore

    cfg = _load_config(args.config)
    family_filter = None
    raw_families = getattr(args, "families", None)
    if raw_families:
        family_filter = tuple(
            family.strip() for family in raw_families.split(",") if family.strip()
        ) or None
    store = HypothesisStore(HYPOTHESES_DB)
    try:
        summary = build_batch_research_summary(
            store.list_observation_active(asset=args.asset),
            top=args.top,
            quality_min=cfg.lifecycle.batch_research_normalized_quality_min,
            families=family_filter,
        )
    finally:
        store.close()
    reasons = summary["reasons"]
    title = f"Batch Research ({args.asset.upper()})"
    if family_filter:
        title += f" [{','.join(family_filter)}]"
    print(title)
    print(
        "  Summary:  "
        f"scored={summary['total']} retained={summary['retained']} "
        f"actionable={summary['actionable']} backed={summary['backed']}"
    )
    print(
        "  Drop:     "
        f"research_q={reasons.get('research_quality', 0)} "
        f"live_q={reasons.get('live_quality', 0)} "
        f"obs={reasons.get('observation', 0)} "
        f"signal={reasons.get('signal', 0)} "
        f"contrib={reasons.get('contribution', 0)} "
        f"candidate_cap={reasons.get('candidate_cap', 0)} "
        f"redundancy={reasons.get('redundancy', 0)} "
        f"other={reasons.get('other', 0)} "
        f"backed={reasons.get('backed', 0)}"
    )
    families = summary["families"]
    family_bits = []
    for reason in (
        "backed",
        "research_quality",
        "live_quality",
        "observation",
        "signal",
        "contribution",
        "candidate_cap",
        "redundancy",
    ):
        values = families.get(reason) or []
        if values:
            family_bits.append(f"{reason}=" + ",".join(values))
    if family_bits:
        print("  Fam:      " + " | ".join(family_bits))
    quality_drop_norm = summary["quality_drop_norm"]
    quality_drop_sharpe = summary["quality_drop_sharpe"]
    quality_drop_folds = summary["quality_drop_folds"]
    backed_norm = summary["backed_norm"]
    print(
        "  Quality:  "
        f"min={summary['quality_threshold']:.2f} "
        f"dropped_p50={quality_drop_norm['p50']:.2f} "
        f"dropped_p90={quality_drop_norm['p90']:.2f} "
        f"dropped_max={quality_drop_norm['max']:.2f} "
        f"backed_p50={backed_norm['p50']:.2f}"
    )
    print(
        "  Inputs:   "
        f"drop_sharpe_p50={quality_drop_sharpe['p50']:.2f} "
        f"drop_sharpe_p90={quality_drop_sharpe['p90']:.2f} "
        f"drop_folds_p50={quality_drop_folds['p50']:.0f}"
    )
    if summary["family_quality_lines"]:
        print("  FamilyQ:  " + " | ".join(summary["family_quality_lines"]))
    if summary["near_miss_entries"]:
        print("  NearMiss: " + ", ".join(summary["near_miss_entries"]))
    for entry in summary["top_entries"]:
        print(f"  Top:      {entry}")


def cmd_analyze_trade_transition(args: argparse.Namespace) -> None:
    from alpha_os.hypotheses.trade_transition_diagnostics import (
        build_trade_transition_summary,
        capture_batch_transition_snapshot,
    )

    cfg = _load_config(args.config)
    asset = args.asset.upper()
    family_filter = None
    raw_families = getattr(args, "families", None)
    if raw_families:
        family_filter = tuple(
            family.strip() for family in raw_families.split(",") if family.strip()
        ) or None

    contexts = _build_trade_contexts(
        asset_list=[asset],
        cfg=cfg,
        testnet=True,
        capital=cfg.trading.initial_capital,
        venue="paper",
    )
    trader, cb, readiness_checker = contexts[asset]
    try:
        pre = capture_batch_transition_snapshot(
            trader.registry.list_observation_active(asset=asset),
            families=family_filter,
        )
        result = trader.run_cycle(skip_lifecycle=cfg.lifecycle_daemon.enabled)
        _print_paper_result(result)
        recon = trader.reconcile()
        _run_trade_readiness_check(result, recon, cb, readiness_checker)
        if not cfg.lifecycle_daemon.enabled:
            _run_hypothesis_lifecycle_update(trader, cfg, result)
        post = capture_batch_transition_snapshot(
            trader.registry.list_observation_active(asset=asset),
            families=family_filter,
        )
    finally:
        _close_trade_contexts(contexts)

    summary = build_trade_transition_summary(pre, post, top=max(args.top, 0))
    title = f"Trade Transition ({asset})"
    if family_filter:
        title += f" [{','.join(family_filter)}]"
    print(title)
    print(
        "  Summary:  "
        f"scoped_pre={summary['scoped_pre']} "
        f"scoped_post={summary['scoped_post']} "
        f"pre_backed={summary['pre_backed']} "
        f"post_backed={summary['post_backed']} "
        f"entered={summary['entered']} "
        f"exited={summary['exited']}"
    )
    if summary["exit_reasons"]:
        print(
            "  Exit:     "
            + " ".join(
                f"{name}={count}"
                for name, count in summary["exit_reasons"].most_common()
            )
        )
    else:
        print("  Exit:     none")
    if summary["entry_reasons"]:
        print(
            "  Enter:    "
            + " ".join(
                f"{name}={count}"
                for name, count in summary["entry_reasons"].most_common()
            )
        )
    else:
        print("  Enter:    none")
    for entry in summary["top_exits"]:
        print(f"  TopExit:  {entry}")
    for entry in summary["top_entries"]:
        print(f"  TopEnter: {entry}")


def cmd_backfill_observation_returns(args: argparse.Namespace) -> None:
    from alpha_os.data.store import DataStore
    from alpha_os.forward.tracker import HypothesisObservationTracker
    from alpha_os.paper.tracker import PaperPortfolioTracker
    from alpha_os.hypotheses import (
        HypothesisStore,
        apply_allocation_rebalance_plan,
        backfill_observation_returns,
        build_capped_allocation_rebalance_plan,
    )

    cfg = _load_config(args.config)
    adir = asset_data_dir(args.asset)
    store = HypothesisStore(HYPOTHESES_DB)
    data_store = DataStore(SIGNAL_CACHE_DB)
    forward_tracker = HypothesisObservationTracker(adir / HYPOTHESIS_OBSERVATIONS_DB_NAME)
    portfolio_tracker = PaperPortfolioTracker(db_path=adir / "paper_trading.db")
    try:
        live_returns_for = _build_live_returns_getter(
            forward_tracker,
            supports_short=cfg.trading.supports_short,
        )
        signal_activity_for = _build_signal_activity_getter(
            portfolio_tracker,
            lookback=cfg.forward.degradation_window,
            supports_short=cfg.trading.supports_short,
        )
        summary = backfill_observation_returns(
            hypothesis_store=store,
            data_store=data_store,
            forward_tracker=forward_tracker,
            asset=args.asset,
            lookback_days=args.days,
        )
        print(
            "Observation backfill: "
            f"asset={args.asset} hypotheses={summary.n_hypotheses} "
            f"days={summary.n_days} records={summary.n_records} "
            f"failures={summary.n_failures}"
        )

        if not args.apply_lifecycle:
            return

        plan = build_capped_allocation_rebalance_plan(
            store,
            asset=args.asset,
            config=cfg,
            live_returns_for=live_returns_for,
            signal_activity_for=signal_activity_for,
            data_store=data_store,
        )
        updates = apply_allocation_rebalance_plan(store, plan)
        print(
            "Lifecycle refresh: "
            f"updated={len(updates)} active={len(plan)} "
            f"capital_backed={sum(1 for entry in plan if entry.proposed_stake > cfg.lifecycle.target_stake_floor)} "
            f"live_proven={sum(1 for entry in plan if entry.live_proven)}"
        )
    finally:
        portfolio_tracker.close()
        forward_tracker.close()
        data_store.close()
        store.close()


def cmd_score_exploratory_hypotheses(args: argparse.Namespace) -> None:
    from alpha_os.hypotheses.research_scoring_service import (
        run_exploratory_research_scoring,
    )
    from alpha_os.hypotheses.store import HypothesisStore

    cfg = _load_config(args.config)
    store = HypothesisStore(HYPOTHESES_DB)
    try:
        run = run_exploratory_research_scoring(
            store=store,
            config=cfg,
            asset=args.asset,
            limit=args.limit,
            dry_run=args.dry_run,
            load_data=_real_data,
        )
        mode = "DRY RUN" if args.dry_run else "APPLY"
        if run.candidate_count == 0:
            print(
                f"Research scoring [{mode}]: asset={args.asset.upper()} "
                "candidates=0 scored=0 failed=0"
            )
            return
        batch = run.batch

        print(
            f"Research scoring [{mode}]: asset={args.asset.upper()} "
            f"candidates={run.candidate_count} scored={len(batch.updates)} failed={len(batch.failures)}"
        )
        for update in batch.updates[:10]:
            print(
                "  "
                f"{update.hypothesis_id}: "
                f"oos_sharpe={update.oos_sharpe:+.3f} "
                f"oos_log_growth={update.oos_log_growth:+.3f} "
                f"folds={update.n_folds}"
            )
        if len(batch.updates) > 10:
            print(f"  ... {len(batch.updates) - 10} more scored hypotheses")
        for failure in batch.failures[:10]:
            print(f"  FAIL {failure.hypothesis_id}: {failure.reason}")
        if len(batch.failures) > 10:
            print(f"  ... {len(batch.failures) - 10} more failures")

        if args.dry_run:
            return
        print(
            "Research scoring summary: "
            f"updated={len(batch.updates)} active={store.count(status='active')}"
        )
    finally:
        store.close()


def cmd_replay_experiment(args: argparse.Namespace) -> None:
    from alpha_os.research.replay_experiment import (
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

    from alpha_os.legacy.admission import AdmissionDaemon

    daemon = AdmissionDaemon(asset=args.asset, config=cfg)
    daemon.run()


def cmd_prune_stale_candidates(args: argparse.Namespace) -> None:
    from alpha_os.legacy.admission_queue import prune_stale_pending_candidates

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


def _hypothesis_status(*, asset: str | None = None) -> dict[str, int]:
    from alpha_os.hypotheses.sleeve_status import build_hypothesis_status_counts
    from alpha_os.hypotheses.store import HypothesisStore

    store = HypothesisStore(HYPOTHESES_DB)
    try:
        counts = build_hypothesis_status_counts(store, asset=asset)
        return {
            "active": counts.active,
            "paused": counts.paused,
            "archived": counts.archived,
            "live": counts.live,
        }
    finally:
        store.close()


def _live_hypothesis_ids(*, asset: str | None = None) -> list[str]:
    from alpha_os.hypotheses.sleeve_status import live_hypothesis_ids
    from alpha_os.hypotheses.store import HypothesisStore

    store = HypothesisStore(HYPOTHESES_DB)
    try:
        return live_hypothesis_ids(store, asset=asset)
    finally:
        store.close()


def _runtime_cohort(record) -> str:
    from alpha_os.hypotheses.sleeve_status import runtime_cohort

    return runtime_cohort(record)


def _runtime_hypothesis_summary(*, asset: str | None = None) -> dict[str, object]:
    from alpha_os.hypotheses.sleeve_status import build_asset_sleeve_summary
    from alpha_os.hypotheses.store import HypothesisStore

    store = HypothesisStore(HYPOTHESES_DB)
    try:
        summary = build_asset_sleeve_summary(store.list_observation_active(asset=asset))
    finally:
        store.close()

    return {
        "bootstrap_backed": summary.bootstrap_backed,
        "observed": summary.observed,
        "capital_backed": summary.capital_backed,
        "research_retained": summary.research_retained,
        "bootstrap_research_retained": summary.bootstrap_research_retained,
        "batch_research_retained": summary.batch_research_retained,
        "live_proven": summary.live_proven,
        "actionable_live": summary.actionable_live,
        "promoted_live": summary.promoted_live,
        "research_demoted": summary.research_demoted,
        "research_candidate_capped": summary.research_candidate_capped,
        "bootstrap_capital_backed": summary.bootstrap_capital_backed,
        "batch_research_capital_backed": summary.batch_research_capital_backed,
        "actionable_live_capital_backed": summary.actionable_live_capital_backed,
        "actionable_redundancy_capped": summary.actionable_redundancy_capped,
        "actionable_other_dropped": summary.actionable_other_dropped,
        "promotion_blockers": summary.promotion_blockers,
        "top_allocation": summary.top_allocation,
        "top_effective_live": summary.top_effective_live,
        "top_raw_live": summary.top_raw_live,
        "top_bootstrap": summary.top_bootstrap,
        "top_actionable_capped": summary.top_actionable_capped,
        "batch_retained_families": summary.batch_retained_families,
        "batch_backed_families": summary.batch_backed_families,
    }


def _runtime_actionable_window_summary(
    *,
    asset: str,
    lookback: int,
    supports_short: bool,
) -> dict[str, float] | None:
    from alpha_os.hypotheses.sleeve_status import build_actionable_window_summary
    from alpha_os.hypotheses.store import HypothesisStore
    from alpha_os.paper.tracker import PaperPortfolioTracker

    tracker = PaperPortfolioTracker(db_path=asset_data_dir(asset) / "paper_trading.db")
    store = HypothesisStore(HYPOTHESES_DB)
    try:
        summary = build_actionable_window_summary(
            store.list_live(asset=asset),
            tracker=tracker,
            lookback=lookback,
            supports_short=supports_short,
        )
        if summary is None:
            return None
        return {
            "lookback": float(summary.lookback),
            "tracked": float(summary.tracked),
            "expressing": float(summary.expressing),
            "mean_ratio": summary.mean_ratio,
            "mean_action": summary.mean_action,
            "breadth": summary.breadth,
        }
    finally:
        store.close()
        tracker.close()


def _runtime_observation_findings(
    latest: dict | None,
    current_live_count: int,
) -> list[str]:
    findings: list[str] = []
    if latest is None:
        findings.append("no readiness reports yet")
        return findings

    if latest.get("has_errors"):
        findings.append("latest cycle has runtime errors")
    if latest.get("n_fills", 0) == 0:
        findings.append("latest cycle had zero fills")
    if latest.get("n_skipped_no_delta", 0) > 0 and latest.get("n_fills", 0) == 0:
        findings.append("latest cycle had no_delta in long-only mode")
    if latest.get("n_skipped_deadband", 0) > 0 and latest.get("n_fills", 0) == 0:
        findings.append("deadband skipped the latest cycle")
    if latest.get("n_order_failures", 0) > 0:
        findings.append("latest cycle had order failures")
    if not latest.get("reconciliation_match", False):
        findings.append("latest cycle failed reconciliation")
    latest_active = int(
        latest.get("n_active_hypotheses", latest.get("n_registry_active", current_live_count))
    )
    if latest_active != current_live_count:
        findings.append("latest report is older than current runtime selection state")
    return findings


def _runtime_observation_verdict(latest: dict | None, findings: list[str]) -> str:
    if latest is None:
        return "pending"
    critical = {
        "latest cycle has runtime errors",
        "latest cycle had order failures",
        "latest cycle failed reconciliation",
    }
    if any(finding in critical for finding in findings):
        return "attention"
    if findings:
        return "watch"
    return "ok"


def _latest_live_count(latest: dict | None) -> int:
    if latest is None:
        return 0
    return int(latest.get("n_live_hypotheses", 0))


def _current_runtime_profile(cfg, asset: str):
    live_ids = _live_hypothesis_ids(asset=asset)
    profile = build_runtime_profile(
        asset=asset,
        config=cfg,
        live_hypothesis_ids=live_ids,
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
    hypotheses = _hypothesis_status(asset=args.asset)
    runtime_ids = _live_hypothesis_ids(asset=args.asset)
    hypothesis_summary = _runtime_hypothesis_summary(asset=args.asset)
    actionable_window = _runtime_actionable_window_summary(
        asset=args.asset,
        lookback=cfg.forward.degradation_window,
        supports_short=cfg.trading.supports_short,
    )
    current_profile = _current_runtime_profile(cfg, args.asset)
    findings = _runtime_observation_findings(latest, len(runtime_ids))

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
        f"  Runtime:   source=hypotheses live={len(runtime_ids)}"
    )
    print(
        f"  Hypotheses: active={hypotheses['active']} paused={hypotheses['paused']} "
        f"archived={hypotheses['archived']} live={hypotheses['live']}"
    )
    print(
        "  Signals:   "
        f"observed={hypothesis_summary['observed']} "
        f"bootstrap_backed={hypothesis_summary['bootstrap_backed']} "
        f"research_retained={hypothesis_summary['research_retained']} "
        f"live_proven={hypothesis_summary['live_proven']} "
        f"actionable_live={hypothesis_summary['actionable_live']} "
        f"promoted_live={hypothesis_summary['promoted_live']} "
        f"research_demoted={hypothesis_summary['research_demoted']} "
        f"research_candidate_capped={hypothesis_summary['research_candidate_capped']} "
        f"capital_backed={hypothesis_summary['capital_backed']}"
    )
    print(
        "  Cohorts:   "
        f"bootstrap={hypothesis_summary['bootstrap_research_retained']}/"
        f"{hypothesis_summary['bootstrap_capital_backed']} "
        f"batch={hypothesis_summary['batch_research_retained']}/"
        f"{hypothesis_summary['batch_research_capital_backed']} "
        f"live={hypothesis_summary['live_proven']}/"
        f"{hypothesis_summary['actionable_live_capital_backed']}"
    )
    if (
        hypothesis_summary["actionable_redundancy_capped"] > 0
        or hypothesis_summary["actionable_other_dropped"] > 0
    ):
        print(
            "  Actionable:"
            f" backed={hypothesis_summary['actionable_live_capital_backed']}"
            f" redundancy_capped={hypothesis_summary['actionable_redundancy_capped']}"
            f" other_dropped={hypothesis_summary['actionable_other_dropped']}"
        )
    if hypothesis_summary["top_actionable_capped"]:
        print("  TopCap:    " + ", ".join(hypothesis_summary["top_actionable_capped"]))
    if (
        hypothesis_summary["batch_retained_families"]
        or hypothesis_summary["batch_backed_families"]
    ):
        retained = ", ".join(hypothesis_summary["batch_retained_families"]) or "-"
        backed = ", ".join(hypothesis_summary["batch_backed_families"]) or "-"
        print(f"  BatchFam:  retained={retained} backed={backed}")
    if actionable_window is not None:
        print(
            "  ActionWin: "
            f"lookback={int(actionable_window['lookback'])} "
            f"tracked={int(actionable_window['tracked'])} "
            f"expressing={int(actionable_window['expressing'])} "
            f"mean_ratio={actionable_window['mean_ratio']:.3f} "
            f"mean_action={actionable_window['mean_action']:.3f} "
            f"breadth={actionable_window['breadth']:.2f}"
        )
    blocker_counts = hypothesis_summary["promotion_blockers"]
    if any(blocker_counts.values()):
        print(
            "  Promote:   "
            f"obs={blocker_counts['insufficient_observations']} "
            f"quality={blocker_counts['weak_live_quality']} "
            f"contrib={blocker_counts['weak_marginal_contribution']} "
            f"both={blocker_counts['weak_live_quality_and_contribution']} "
            f"signal={blocker_counts['weak_signal_activity']}"
        )
    if hypothesis_summary["top_allocation"]:
        print("  TopAlloc:  " + ", ".join(hypothesis_summary["top_allocation"]))
    if hypothesis_summary["top_effective_live"]:
        print("  TopEff:    " + ", ".join(hypothesis_summary["top_effective_live"]))
    if hypothesis_summary["top_raw_live"]:
        print("  TopRaw:    " + ", ".join(hypothesis_summary["top_raw_live"]))
    if hypothesis_summary["top_bootstrap"]:
        print("  TopBoot:   " + ", ".join(hypothesis_summary["top_bootstrap"]))
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
        latest_live_set_id = latest.get("profile_live_set_id", "")
        if latest_config_id or latest_live_set_id:
            print(
                "  ProfileIDs: "
                f"config={latest_config_id[:12] or 'n/a'} "
                f"live={latest_live_set_id[:12] or 'n/a'}"
            )
            print(
                "  CurrentIDs: "
                f"config={current_profile.config_id[:12]} "
                f"live={current_profile.live_set_id[:12]}"
            )
    print(
        f"  Selection: current_live={len(runtime_ids)} "
        f"latest_active={latest.get('n_active_hypotheses', latest.get('n_registry_active', 0))} "
        f"live={_latest_live_count(latest)} "
        f"selected={latest.get('n_selected_hypotheses', latest.get('n_selected_alphas', 0))}"
    )
    print(
        "  Skips:     "
        f"deadband={latest['n_skipped_deadband']} "
        f"no_delta={latest.get('n_skipped_no_delta', 0)} "
        f"min_notional={latest['n_skipped_min_notional']} "
        f"rounded_to_zero={latest['n_skipped_rounded_to_zero']}"
    )
    print(
        f"  Health:    reconciliation={latest['reconciliation_match']} "
        f"order_failures={latest['n_order_failures']} "
        f"halted={latest['circuit_breaker_halted']}"
    )
    verdict = _runtime_observation_verdict(latest, findings)
    print(f"  Observe:   {verdict}")
    for finding in findings:
        print(f"    - {finding}")
    if latest_profile_id and latest_profile_id != current_profile.profile_id:
        print("    - latest report was recorded under a different runtime profile")
        latest_config_id = latest.get("profile_config_id", "")
        latest_live_set_id = latest.get("profile_live_set_id", "")
        if latest_config_id and latest_config_id != current_profile.config_id:
            print("    - config fingerprint differs between current and latest")
        if latest_live_set_id and latest_live_set_id != current_profile.live_set_id:
            print("    - live hypothesis set fingerprint differs between current and latest")
    latest_active = int(latest.get("n_active_hypotheses", latest.get("n_registry_active", 0)))
    if latest_active != len(runtime_ids):
        print(
            "  Note:      current runtime live count differs from latest readiness report; "
            "the next trade cycle will refresh the report."
        )


def cmd_analyze_latest_combine(args: argparse.Namespace) -> None:
    from alpha_os.hypotheses.combiner import compute_stake_weights
    from alpha_os.hypotheses.sleeve_status import build_latest_combine_summary
    from alpha_os.hypotheses.store import HypothesisStore
    from alpha_os.paper.tracker import PaperPortfolioTracker

    tracker = PaperPortfolioTracker(db_path=asset_data_dir(args.asset) / "paper_trading.db")
    store = HypothesisStore(HYPOTHESES_DB)
    try:
        snapshot = tracker.get_last_snapshot()
        print(f"Latest Combine ({args.asset.upper()})")
        if snapshot is None:
            print("  No portfolio snapshots yet.")
            return

        signals = tracker.get_hypothesis_signals(snapshot.date)
        if not signals:
            print(f"  No hypothesis signals saved for snapshot {snapshot.date}.")
            return

        record_map = {
            record.hypothesis_id: record
            for record in store.list_observation_active(asset=args.asset)
        }
        stakes = {
            hypothesis_id: float(record_map[hypothesis_id].stake)
            for hypothesis_id in signals
            if hypothesis_id in record_map and float(record_map[hypothesis_id].stake) > 0
        }
        weights = compute_stake_weights(stakes)
        summary = build_latest_combine_summary(
            record_map=record_map,
            signals=signals,
            weights=weights,
            top_n=max(int(args.top), 1),
        )
        print(f"  Date:      {snapshot.date}")
        print(
            f"  Combined:  stored={float(snapshot.combined_signal):+.6f} "
            f"current_weighted={summary.current_combined:+.6f}"
        )
        print(
            f"  Snapshot:  selected={len(signals)} "
            f"current_backed={summary.current_backed} "
            f"dropped={summary.dropped_current} missing={summary.missing_current}"
        )
        print(
            f"  Current:   nonzero={summary.nonzero_current} "
            f"zero={summary.zero_current}"
        )
        if summary.dropped_reasons:
            parts = [f"{key}={value}" for key, value in sorted(summary.dropped_reasons.items())]
            print(f"  Dropped:   {' '.join(parts)}")
        print(
            "  Cohorts:   "
            f"bootstrap n={summary.cohorts['bootstrap'].n}/{summary.cohorts['bootstrap'].nonzero} "
            f"w={summary.cohorts['bootstrap'].weight:.3f} "
            f"sig={summary.cohorts['bootstrap'].weighted_signal:+.6f} | "
            f"batch n={summary.cohorts['batch'].n}/{summary.cohorts['batch'].nonzero} "
            f"w={summary.cohorts['batch'].weight:.3f} "
            f"sig={summary.cohorts['batch'].weighted_signal:+.6f} | "
            f"live n={summary.cohorts['live'].n}/{summary.cohorts['live'].nonzero} "
            f"w={summary.cohorts['live'].weight:.3f} "
            f"sig={summary.cohorts['live'].weighted_signal:+.6f}"
        )
        for entry in summary.top_entries:
            print(
                "  Top:       "
                f"{entry.hypothesis_id} cohort={entry.cohort} weight={entry.weight:.4f} "
                f"signal={entry.signal:+.4f} contrib={entry.contribution:+.6f}"
            )
    finally:
        store.close()
        tracker.close()


def _resolve_signal_cache_targets(args: argparse.Namespace) -> list[str]:
    from alpha_os.data.universe import price_signal

    assets = [item.strip().upper() for item in (args.assets or args.asset).split(",") if item.strip()]
    signals: set[str] = set()

    if args.signals:
        signals.update(item.strip() for item in args.signals.split(",") if item.strip())
    else:
        for asset in assets:
            signals.add(price_signal(asset))

    if args.from_hypotheses:
        from alpha_os.hypotheses import HypothesisStore
        from alpha_os.hypotheses.producer import collect_required_features
        from alpha_os.hypotheses.sleeve_scope import filter_records_by_assets

        store = HypothesisStore(HYPOTHESES_DB)
        try:
            active = filter_records_by_assets(store.list_active(), assets)
        finally:
            store.close()
        signals.update(collect_required_features(active, assets))

    return sorted(signals)


def cmd_sync_signal_cache(args: argparse.Namespace) -> None:
    from alpha_os.data.store import DataStore
    from alpha_os.hypotheses.producer import _quick_healthcheck

    asset_list = [item.strip().upper() for item in (args.assets or args.asset).split(",") if item.strip()]
    lock_path = runtime_lock_path("sync-signal-cache", asset_list)
    try:
        runtime_lock = hold_runtime_lock(lock_path)
        runtime_lock.__enter__()
    except RuntimeLockBusy:
        print(
            "Signal cache sync already active for "
            f"{','.join(asset_list)}; skipping overlapping invocation."
        )
        print(f"Sync summary: assets={','.join(asset_list)} status=skipped_overlap")
        if args.strict:
            sys.exit(1)
        return

    cfg = Config.load(args.config)
    try:
        targets = _resolve_signal_cache_targets(args)
        if not targets:
            print("No signals selected for sync")
            print(f"Sync summary: assets={','.join(asset_list)} status=skipped_no_targets")
            if args.strict:
                sys.exit(1)
            return

        db_path = SIGNAL_CACHE_L2_DB if args.resolution != "1d" else SIGNAL_CACHE_DB
        client = build_signal_client_from_config(cfg.api)

        if not _quick_healthcheck(cfg.api.base_url):
            print(f"signal-noise unavailable at {cfg.api.base_url} — skipping sync")
            print(
                f"Sync summary: assets={','.join(asset_list)} "
                "status=skipped_source_unavailable"
            )
            if args.strict:
                sys.exit(1)
            return

        store = DataStore(db_path, client)
        before = store.signal_row_counts(targets)
        try:
            store.sync(
                targets,
                resolution=args.resolution,
                min_history_days=max(args.min_history_days, 0),
            )
            after = store.signal_row_counts(targets)
        finally:
            store.close()

        populated = sum(1 for signal in targets if after.get(signal, 0) > 0)
        improved = sum(1 for signal in targets if after.get(signal, 0) > before.get(signal, 0))
        print(
            f"Signal cache sync ({args.resolution}) "
            f"targets={len(targets)} populated={populated} improved={improved}"
        )
        for signal in targets[:20]:
            print(f"  {signal}: {before.get(signal, 0)} -> {after.get(signal, 0)}")
        if len(targets) > 20:
            print(f"  ... {len(targets) - 20} more")
        if args.strict and populated < len(targets):
            print(
                f"Strict mode: {len(targets) - populated} signals remain empty after sync"
            )
            print(
                "Sync summary: "
                f"assets={','.join(asset_list)} status=incomplete "
                f"targets={len(targets)} populated={populated} improved={improved}"
            )
            sys.exit(1)
        print(
            "Sync summary: "
            f"assets={','.join(asset_list)} status=ok "
            f"targets={len(targets)} populated={populated} improved={improved}"
        )
    finally:
        runtime_lock.__exit__(None, None, None)


def cmd_alpha_funnel(args: argparse.Namespace) -> None:
    from alpha_os.legacy.funnel import load_funnel_summary

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
    from alpha_os.config import Config
    from alpha_os.data.store import DataStore
    from alpha_os.data.signal_client import build_signal_client_from_config
    from alpha_os.data.universe import init_universe, load_daily_signals, required_raw_signals
    from alpha_os.data.eval_universe import load_cached_eval_universe
    from alpha_os.hypotheses.identity import expression_feature_names
    from alpha_os.research.cross_asset import (
        DEFAULT_HORIZONS,
        evaluate_cross_asset_multi_horizon,
    )

    cfg = Config.load(args.config)
    client = build_signal_client_from_config(cfg.api)
    init_universe(client)
    all_signals = load_daily_signals(client)

    # Load cached eval universe
    eval_assets = load_cached_eval_universe()
    if not eval_assets:
        print("No cached eval universe. Run unified-generator first.")
        return

    # Load data: eval universe prices + features referenced by expression
    expr_features = required_raw_signals(expression_feature_names(args.expr))
    db_path = SIGNAL_CACHE_DB
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
        print("\nPer-asset IC (top 10):")
        sorted_assets = sorted(result.per_asset.items(), key=lambda x: x[1], reverse=True)
        for asset, ic in sorted_assets[:10]:
            print(f"  {asset:30s} IC={ic:+.4f}")


def cmd_produce_classical(args: argparse.Namespace) -> None:
    """Run classical indicator producer → prediction store."""
    from alpha_os.config import Config
    from alpha_os.predictions.classical_producer import produce_classical_predictions

    cfg = Config.load(args.config)
    n = produce_classical_predictions(cfg)
    print(f"Wrote {n} classical predictions to store")


def cmd_produce_predictions(args: argparse.Namespace) -> None:
    """Run the active signal producer → prediction store."""
    from alpha_os.config import Config
    from alpha_os.hypotheses.producer import produce_active_hypothesis_predictions

    asset_list = [args.asset.upper()]
    lock_path = runtime_lock_path("produce-predictions", asset_list)
    try:
        runtime_lock = hold_runtime_lock(lock_path)
        runtime_lock.__enter__()
    except RuntimeLockBusy:
        print(
            "Prediction production already active for "
            f"{','.join(asset_list)}; skipping overlapping invocation."
        )
        print(f"Prediction summary: asset={asset_list[0]} status=skipped_overlap")
        if args.strict:
            sys.exit(1)
        return

    try:
        cfg = Config.load(args.config)
        n = produce_active_hypothesis_predictions(cfg, assets=[args.asset])
        print(f"Wrote {n} hypothesis predictions to store")
        if args.strict and n <= 0:
            print("Strict mode: no hypothesis predictions were written")
            print(f"Prediction summary: asset={asset_list[0]} status=empty written={n}")
            sys.exit(1)
        status = "ok" if n > 0 else "empty"
        print(f"Prediction summary: asset={asset_list[0]} status={status} written={n}")
    finally:
        runtime_lock.__exit__(None, None, None)


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
    elif args.command == "hypothesis-seeder":
        cmd_hypothesis_seeder(args)
    elif args.command == "score-exploratory-hypotheses":
        cmd_score_exploratory_hypotheses(args)
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
    elif args.command == "rebalance-allocation-trust":
        cmd_rebalance_allocation_trust(args)
    elif args.command == "analyze-live-breadth":
        cmd_analyze_live_breadth(args)
    elif args.command == "analyze-batch-research":
        cmd_analyze_batch_research(args)
    elif args.command == "analyze-trade-transition":
        cmd_analyze_trade_transition(args)
    elif args.command == "analyze-latest-combine":
        cmd_analyze_latest_combine(args)
    elif args.command == "backfill-observation-returns":
        cmd_backfill_observation_returns(args)
    elif args.command == "replay-experiment":
        cmd_replay_experiment(args)
    elif args.command == "replay-matrix":
        cmd_replay_matrix(args)
    elif args.command == "testnet-readiness":
        cmd_testnet_readiness(args)
    elif args.command == "runtime-status":
        cmd_runtime_status(args)
    elif args.command == "sync-signal-cache":
        cmd_sync_signal_cache(args)
    elif args.command == "alpha-funnel":
        cmd_alpha_funnel(args)
    elif args.command == "legacy":
        if args.legacy_command == "unified-generator":
            cmd_unified_generator(args)
        elif args.legacy_command == "enqueue-discovery-pool":
            cmd_enqueue_discovery_pool(args)
        elif args.legacy_command == "admission-daemon":
            cmd_admission_daemon(args)
        elif args.legacy_command == "prune-stale-candidates":
            cmd_prune_stale_candidates(args)
        elif args.legacy_command == "lifecycle":
            cmd_lifecycle(args)
        elif args.legacy_command == "alpha-funnel":
            cmd_alpha_funnel(args)
    elif args.command == "research":
        if args.research_command == "generate":
            cmd_generate(args)
        elif args.research_command == "backtest":
            cmd_backtest(args)
        elif args.research_command == "evolve":
            cmd_evolve(args)
        elif args.research_command == "validate":
            cmd_validate(args)
        elif args.research_command == "evaluate":
            cmd_evaluate_expression(args)
        elif args.research_command == "produce-classical":
            cmd_produce_classical(args)
        elif args.research_command == "paper-replay":
            cmd_paper_replay(args)
        elif args.research_command == "replay-experiment":
            cmd_replay_experiment(args)
        elif args.research_command == "replay-matrix":
            cmd_replay_matrix(args)
    elif args.command == "evaluate":
        cmd_evaluate_expression(args)
    elif args.command == "produce-predictions":
        cmd_produce_predictions(args)
    elif args.command == "produce-classical":
        cmd_produce_classical(args)
