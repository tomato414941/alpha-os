#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from alpha_os.alpha.deployed_alphas import refresh_deployed_alphas
from alpha_os.alpha.handcrafted import get_handcrafted_expressions
from alpha_os.alpha.registry import AlphaRegistry, AlphaState
from alpha_os.config import Config, asset_data_dir
from alpha_os.paper.simulator import run_replay


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated handcrafted-baseline replay against the current profile.",
    )
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--alpha-set", default="baseline")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--config", default=None)
    return parser


def _run_handcrafted_replay(asset: str, alpha_set: str, cfg: Config, start: str, end: str) -> None:
    import alpha_os.config as config_mod
    import alpha_os.daemon.admission as admission_mod

    asset = asset.upper()
    current = run_replay(
        asset=asset,
        config=cfg,
        start_date=start,
        end_date=end,
        sizing_mode="runtime",
    )

    expressions = get_handcrafted_expressions(asset, alpha_set)
    with tempfile.TemporaryDirectory(prefix="alpha_os_handcrafted_") as tmp:
        data_root = Path(tmp)
        db_path = data_root / asset / "alpha_registry.db"
        registry = AlphaRegistry(db_path)
        try:
            registry.queue_candidate_expressions(
                expressions,
                source=f"handcrafted_{alpha_set}",
                fitness=0.0,
                behavior_json={"source": "handcrafted", "set": alpha_set, "asset": asset},
            )
        finally:
            registry.close()

        old_config_data = config_mod.DATA_DIR
        old_admission_data = admission_mod.DATA_DIR
        try:
            config_mod.DATA_DIR = data_root
            admission_mod.DATA_DIR = data_root

            daemon = admission_mod.AdmissionDaemon(asset=asset, config=cfg)
            daemon.admission_cfg.batch_size = max(len(expressions), daemon.admission_cfg.batch_size)
            daemon.admission_cfg.min_queue_size = 1
            daemon.admission_cfg.max_active_alphas = 0
            daemon._run_batch()
        finally:
            config_mod.DATA_DIR = old_config_data
            admission_mod.DATA_DIR = old_admission_data

        registry = AlphaRegistry(db_path)
        try:
            active_count = registry.count(AlphaState.ACTIVE)
            rejected_count = registry.count(AlphaState.REJECTED)
            adopted = registry.list_active()
        finally:
            registry.close()

        refresh_deployed_alphas(
            db_path,
            cfg,
            forward_db_path=asset_data_dir(asset) / "forward_returns.db",
            dry_run=False,
            backup=False,
        )

        handcrafted = run_replay(
            asset=asset,
            config=cfg,
            start_date=start,
            end_date=end,
            registry_db=db_path,
            sizing_mode="runtime",
        )

    print(f"Hand-crafted replay ({asset}:{alpha_set})")
    print(f"  Candidate expressions: {len(expressions)}")
    print(f"  Adopted:               {active_count}")
    print(f"  Rejected:              {rejected_count}")
    if adopted:
        print("  Active alpha ids:      " + ", ".join(record.alpha_id for record in adopted))
    print("")
    print("Current profile")
    print(
        f"  Return={current.total_return:+.2%} Sharpe={current.sharpe:.3f} "
        f"DD={current.max_drawdown:.2%} Trades={current.total_trades}"
    )
    print("Hand-crafted only")
    print(
        f"  Return={handcrafted.total_return:+.2%} Sharpe={handcrafted.sharpe:.3f} "
        f"DD={handcrafted.max_drawdown:.2%} Trades={handcrafted.total_trades}"
    )
    print("Delta (handcrafted - current)")
    print(
        f"  Return={handcrafted.total_return - current.total_return:+.2%} "
        f"Sharpe={handcrafted.sharpe - current.sharpe:+.3f} "
        f"DD={handcrafted.max_drawdown - current.max_drawdown:+.2%} "
        f"Trades={handcrafted.total_trades - current.total_trades:+d}"
    )


def main() -> None:
    args = _build_parser().parse_args()
    cfg = Config.load(Path(args.config)) if args.config else Config.load()
    _run_handcrafted_replay(args.asset, args.alpha_set, cfg, args.start, args.end)


if __name__ == "__main__":
    main()
