"""Tests for the trade CLI command, Trader rename, and Phase 4 features."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_trader_alias():
    """PaperTrader alias still works after rename."""
    from alpha_os.paper.trader import Trader, PaperTrader
    assert PaperTrader is Trader


def test_executor_abc_has_defaults():
    """Executor ABC provides default implementations for optional methods."""
    from alpha_os.execution.executor import Executor

    class MinimalExecutor(Executor):
        def submit_order(self, order):
            return None

        def get_position(self, symbol):
            return 0.0

        def get_cash(self):
            return 1000.0

    ex = MinimalExecutor()
    ex.set_price("BTC", 50000.0)  # no-op, should not raise
    assert ex.portfolio_value == 1000.0  # defaults to get_cash()
    assert ex.all_positions == {}
    assert ex.all_fills == []
    assert ex.get_exchange_position("BTC") == 0.0
    assert ex.get_exchange_cash() == 1000.0
    assert ex.get_reconciled_position("BTC") == 0.0
    assert ex.get_reconciled_cash() == 1000.0


def test_binance_executor_portfolio_value():
    """BinanceExecutor.portfolio_value uses managed state, not exchange balance."""
    from alpha_os.execution.binance import BinanceExecutor

    mock_ex = MagicMock()
    mock_ex.fetch_ticker.return_value = {"last": 50000.0}

    executor = object.__new__(BinanceExecutor)
    executor._exchange = mock_ex
    executor._testnet = True
    executor._symbol_map = {}
    executor._max_slippage_bps = 10.0
    executor._max_book_fraction = 0.1
    executor._managed_cash = 5000.0
    executor._managed_positions = {"BTC": 0.1}
    executor._initial_capital = 10000.0

    assert executor.portfolio_value == pytest.approx(10000.0)  # 5000 + 0.1*50000


def test_binance_executor_all_positions():
    """BinanceExecutor.all_positions returns managed positions only."""
    from alpha_os.execution.binance import BinanceExecutor

    executor = object.__new__(BinanceExecutor)
    executor._exchange = MagicMock()
    executor._testnet = True
    executor._symbol_map = {}
    executor._max_slippage_bps = 10.0
    executor._max_book_fraction = 0.1
    executor._managed_cash = 5000.0
    executor._managed_positions = {"BTC": 0.1, "ETH": 0.0}
    executor._initial_capital = 10000.0

    positions = executor.all_positions
    assert "BTC" in positions
    assert positions["BTC"] == 0.1
    assert "ETH" not in positions  # zero position filtered


def test_trade_parser():
    """CLI parser accepts trade subcommand with correct defaults."""
    from alpha_os.cli import _build_parser

    parser = _build_parser()

    # Testnet (default)
    args = parser.parse_args(["trade", "--once"])
    assert args.command == "trade"
    assert args.once is True
    assert args.real is False
    assert args.capital == 10000.0
    assert args.asset == "BTC"
    assert not hasattr(args, "tactical")

    # Real mode
    args = parser.parse_args([
        "trade", "--once", "--real", "--capital", "500",
    ])
    assert args.real is True
    assert args.capital == 500.0


def test_monitor_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["monitor", "--once", "--asset", "BTC"])

    assert args.command == "monitor"
    assert args.once is True
    assert args.asset == "BTC"


def test_paper_replay_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "paper", "--replay", "--start", "2025-09-01", "--end", "2026-03-05",
    ])

    assert args.command == "paper"
    assert args.replay is True
    assert args.sizing_mode == "runtime"


def test_legacy_user_facing_commands_are_rejected():
    from alpha_os.cli import _build_parser

    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["live", "--once"])
    with pytest.raises(SystemExit):
        parser.parse_args(["forward", "--once"])
    with pytest.raises(SystemExit):
        parser.parse_args(["validator"])
    with pytest.raises(SystemExit):
        parser.parse_args(["validate-testnet"])


def test_rebuild_managed_alphas_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "rebuild-managed-alphas",
        "--asset", "BTC",
        "--source", "candidates",
        "--fail-state", "dormant",
        "--dry-run",
    ])

    assert args.command == "rebuild-managed-alphas"
    assert args.asset == "BTC"
    assert args.source == "candidates"
    assert args.fail_state == "dormant"
    assert args.dry_run is True


def test_refresh_deployed_alphas_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "refresh-deployed-alphas",
        "--asset", "BTC",
        "--dry-run",
    ])

    assert args.command == "refresh-deployed-alphas"
    assert args.asset == "BTC"
    assert args.dry_run is True


def test_prune_managed_alpha_duplicates_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "prune-managed-alpha-duplicates",
        "--asset", "BTC",
        "--dry-run",
        "--no-refresh-deployed",
    ])

    assert args.command == "prune-managed-alpha-duplicates"
    assert args.asset == "BTC"
    assert args.dry_run is True
    assert args.no_refresh_deployed is True


def test_runtime_status_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "runtime-status",
        "--asset", "BTC",
    ])

    assert args.command == "runtime-status"
    assert args.asset == "BTC"


def test_alpha_funnel_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "alpha-funnel",
        "--asset", "BTC",
    ])

    assert args.command == "alpha-funnel"
    assert args.asset == "BTC"


def test_promote_discovery_pool_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "promote-discovery-pool",
        "--asset", "BTC",
        "--limit", "12",
        "--dry-run",
    ])

    assert args.command == "promote-discovery-pool"
    assert args.asset == "BTC"
    assert args.limit == 12
    assert args.dry_run is True


def test_seed_handcrafted_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "seed-handcrafted",
        "--asset", "BTC",
        "--alpha-set", "baseline",
        "--dry-run",
    ])

    assert args.command == "seed-handcrafted"
    assert args.asset == "BTC"
    assert args.alpha_set == "baseline"
    assert args.dry_run is True


def test_analyze_diversity_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "analyze-diversity",
        "--asset", "BTC",
        "--scope", "active",
        "--limit", "50",
        "--metric", "log_growth",
        "--lookback", "126",
        "--top-pairs", "5",
        "--json",
    ])

    assert args.command == "analyze-diversity"
    assert args.asset == "BTC"
    assert args.scope == "active"
    assert args.limit == 50
    assert args.metric == "log_growth"
    assert args.lookback == 126
    assert args.top_pairs == 5
    assert args.json is True


def test_replay_experiment_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "replay-experiment",
        "--name", "confidence sweep",
        "--start", "2026-02-20",
        "--end", "2026-03-05",
        "--managed-alpha-mode", "admission",
        "--source", "candidates",
        "--deployment-mode", "refresh",
        "--set", "lifecycle.candidate_quality_min=1.10",
        "--set", "live_quality.weight_confidence_floor=0.25",
    ])

    assert args.command == "replay-experiment"
    assert args.name == "confidence sweep"
    assert args.managed_alpha_mode == "admission"
    assert args.source == "candidates"
    assert args.deployment_mode == "refresh"
    assert args.set == [
        "lifecycle.candidate_quality_min=1.10",
        "live_quality.weight_confidence_floor=0.25",
    ]


def test_replay_matrix_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "replay-matrix",
        "--manifest", "experiments/deadband.toml",
        "--max-workers", "4",
    ])

    assert args.command == "replay-matrix"
    assert args.manifest == "experiments/deadband.toml"
    assert args.max_workers == 4


def test_cmd_replay_experiment_prints_profile(capsys, monkeypatch):
    from argparse import Namespace

    from alpha_os.cli import cmd_replay_experiment
    from alpha_os.experiments.replay import ReplayExperimentRun

    monkeypatch.setattr(
        "alpha_os.experiments.replay.run_replay_experiment",
        lambda spec: ReplayExperimentRun(
            experiment_id="exp-1",
            detail_path=Path("/tmp/detail.json"),
            index_path=Path("/tmp/index.jsonl"),
            payload={
                "deployment": {"mode": "refresh"},
                "runtime_profile": {
                    "profile_id": "abcdef1234567890",
                    "git_commit": "deadbeefcafebabe",
                },
                "result": {
                    "final_value": 10100.0,
                    "total_return": 0.01,
                    "sharpe": 1.2,
                    "max_drawdown": 0.03,
                    "total_trades": 7,
                },
            },
        ),
    )

    cmd_replay_experiment(
        Namespace(
            name="smoke",
            asset="BTC",
            start="2026-03-01",
            end="2026-03-05",
            config=None,
            managed_alpha_mode="current",
            source="candidates",
            fail_state="rejected",
            deployment_mode="refresh",
            sizing_mode="runtime",
            set=[],
            notes="",
        )
    )
    output = capsys.readouterr().out
    assert "Profile:  abcdef123456" in output


def test_cmd_replay_matrix_prints_profiles(capsys, monkeypatch):
    from argparse import Namespace

    from alpha_os.cli import cmd_replay_matrix
    from alpha_os.experiments.matrix import ReplayMatrixSpec
    from alpha_os.experiments.replay import ReplayExperimentRun, ReplayExperimentSpec

    monkeypatch.setattr(
        "alpha_os.experiments.matrix.load_replay_matrix",
        lambda path: ReplayMatrixSpec(
            defaults={},
            experiments=[
                ReplayExperimentSpec(
                    name="exp-a",
                    asset="BTC",
                    start_date="2026-03-01",
                    end_date="2026-03-05",
                ),
                ReplayExperimentSpec(
                    name="exp-b",
                    asset="BTC",
                    start_date="2026-03-01",
                    end_date="2026-03-05",
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        "alpha_os.experiments.matrix.run_replay_matrix",
        lambda matrix, max_workers: [
            ReplayExperimentRun(
                experiment_id="exp-a",
                detail_path=Path("/tmp/a.json"),
                index_path=Path("/tmp/index.jsonl"),
                payload={
                    "name": "exp-a",
                    "runtime_profile": {"profile_id": "aaaaabbbbbcccc"},
                    "result": {
                        "total_return": 0.01,
                        "sharpe": 1.0,
                        "max_drawdown": 0.02,
                        "total_trades": 5,
                    },
                },
            ),
            ReplayExperimentRun(
                experiment_id="exp-b",
                detail_path=Path("/tmp/b.json"),
                index_path=Path("/tmp/index.jsonl"),
                payload={
                    "name": "exp-b",
                    "runtime_profile": {"profile_id": "dddddeeeeeffff"},
                    "result": {
                        "total_return": 0.02,
                        "sharpe": 1.1,
                        "max_drawdown": 0.03,
                        "total_trades": 6,
                    },
                },
            ),
        ],
    )

    cmd_replay_matrix(Namespace(manifest="experiments/demo.toml", max_workers=2))
    output = capsys.readouterr().out
    assert "profile=aaaaabbbbbcc" in output
    assert "profile=dddddeeeeeff" in output
    assert "Profiles: 2 unique across 2 runs" in output


def test_normalize_trade_config_preserves_requested_profile():
    from alpha_os.cli import _normalize_trade_config
    from alpha_os.config import Config

    cfg = Config.load()
    cfg.paper.combine_mode = "map_elites"
    cfg.regime.enabled = True

    changes = _normalize_trade_config(cfg)

    assert cfg.paper.combine_mode == "map_elites"
    assert cfg.regime.enabled is True
    assert changes == []


def test_load_runtime_observation_config_prefers_user_prod(tmp_path, monkeypatch):
    from alpha_os.cli import _load_runtime_observation_config

    home = tmp_path / "home"
    prod = home / ".config" / "alpha-os" / "prod.toml"
    prod.parent.mkdir(parents=True, exist_ok=True)
    prod.write_text("[deployment]\nmax_alphas = 150\n")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)

    cfg = _load_runtime_observation_config(None)

    assert cfg.deployment.max_alphas == 150


def test_build_tactical_trader_respects_enable_flag(monkeypatch):
    """Layer 2 trader should only be built when explicitly enabled."""
    from alpha_os import cli
    from alpha_os.config import Config

    created: list[tuple[str, Config]] = []

    class DummyTacticalTrader:
        def __init__(self, asset, config):
            created.append((asset, config))

    monkeypatch.setattr(
        "alpha_os.paper.tactical.TacticalTrader",
        DummyTacticalTrader,
    )

    cfg = Config.load()
    assert cli._build_tactical_trader("BTC", cfg, enabled=False) is None

    tactical = cli._build_tactical_trader("BTC", cfg, enabled=True)
    assert isinstance(tactical, DummyTacticalTrader)
    assert created == [("BTC", cfg)]


def test_print_paper_result_shows_signal_stages(capsys):
    """CLI output should show raw and stage-adjusted signals."""
    from alpha_os.cli import _print_paper_result
    from alpha_os.paper.trader import PaperCycleResult

    result = PaperCycleResult(
        date="2026-03-05T21:52:58",
        combined_signal=0.3005,
        fills=[],
        portfolio_value=9976.90,
        daily_pnl=0.65,
        daily_return=0.00006,
        dd_scale=1.0,
        vol_scale=1.0,
        n_registry_active=615,
        n_deployed_alphas=150,
        n_shortlist_candidates=150,
        n_selected_alphas=30,
        n_signals_evaluated=150,
        strategic_signal=0.4210,
        regime_adjusted_signal=0.4210,
        tactical_adjusted_signal=0.5420,
        final_signal=0.5420,
    )

    _print_paper_result(result)
    output = capsys.readouterr().out

    assert "Signal Raw: +0.3005" in output
    assert "Signal L3:  +0.4210" in output
    assert "Signal Reg: +0.4210" in output
    assert "Signal L2:  +0.5420" in output
    assert "Signal Fin: +0.5420" in output
    assert "Managed:    615 active" in output
    assert "Deployed:   150 alphas" in output
    assert "Shortlist:  150 candidates" in output
    assert "Selected:   30 alphas" in output
    assert "Signals:    150 evaluated" in output


def test_print_paper_result_shows_skip_metrics(capsys):
    from alpha_os.cli import _print_paper_result
    from alpha_os.paper.trader import PaperCycleResult

    result = PaperCycleResult(
        date="2026-03-05T21:52:58",
        combined_signal=0.0,
        fills=[],
        portfolio_value=10000.0,
        daily_pnl=0.0,
        daily_return=0.0,
        dd_scale=1.0,
        vol_scale=1.0,
        n_registry_active=615,
        n_deployed_alphas=30,
        n_shortlist_candidates=30,
        n_selected_alphas=30,
        n_signals_evaluated=30,
        n_skipped_deadband=1,
        n_skipped_min_notional=2,
        n_skipped_rounded_to_zero=3,
    )

    _print_paper_result(result)
    output = capsys.readouterr().out

    assert "Skips:      deadband=1" in output
    assert "Skips:      min_notional=2" in output
    assert "Skips:      rounded_to_zero=3" in output


# ---------------------------------------------------------------------------
# Position display filtering
# ---------------------------------------------------------------------------

def test_print_status_filters_positions(capsys):
    """print_status shows only the traded asset, hides others."""
    from alpha_os.paper.trader import Trader
    from alpha_os.paper.tracker import PaperTradingSummary

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    summary = PaperTradingSummary(
        start_date="2026-02-20",
        end_date="2026-02-27",
        n_days=7,
        initial_value=10000.0,
        final_value=10500.0,
        total_return=0.05,
        sharpe=1.5,
        max_drawdown=0.02,
        total_trades=5,
        current_positions={
            "BTC": 0.1,
            "ETH": 1.0,
            "SOL": 5.0,
            "DOGE": 10000.0,
        },
        current_cash=5000.0,
    )
    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.summary.return_value = summary

    trader.print_status()
    output = capsys.readouterr().out

    assert "BTC: 0.100000" in output
    assert "ETH" not in output
    assert "SOL" not in output
    assert "DOGE" not in output
    assert "3 other positions hidden" in output


def test_print_status_no_positions(capsys):
    """print_status handles empty positions."""
    from alpha_os.paper.trader import Trader
    from alpha_os.paper.tracker import PaperTradingSummary

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    summary = PaperTradingSummary(
        start_date="2026-02-27",
        end_date="2026-02-27",
        n_days=1,
        initial_value=10000.0,
        final_value=10000.0,
        total_return=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        total_trades=0,
        current_positions={},
        current_cash=10000.0,
    )
    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.summary.return_value = summary

    trader.print_status()
    output = capsys.readouterr().out

    assert "hidden" not in output


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def test_reconcile_match():
    """reconcile() reports match when internal == exchange."""
    from alpha_os.paper.trader import Trader
    from alpha_os.paper.tracker import PortfolioSnapshot

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    snapshot = PortfolioSnapshot(
        date="2026-02-27",
        cash=5000.0,
        positions={"btc_ohlcv": 0.1},
        portfolio_value=10000.0,
        daily_pnl=100.0,
        daily_return=0.01,
        combined_signal=0.5,
        dd_scale=1.0,
        vol_scale=1.0,
    )
    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.get_last_snapshot.return_value = snapshot

    trader.executor = MagicMock()
    trader.executor.get_reconciled_position.return_value = 0.1
    trader.executor.get_reconciled_cash.return_value = 5000.0

    result = trader.reconcile()
    assert result["match"] is True
    assert result["internal_qty"] == pytest.approx(0.1)
    assert result["qty_diff"] < 1e-6
    assert result["cash_diff"] < 1.0


def test_reconcile_mismatch():
    """reconcile() detects position mismatch."""
    from alpha_os.paper.trader import Trader
    from alpha_os.paper.tracker import PortfolioSnapshot

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    snapshot = PortfolioSnapshot(
        date="2026-02-27",
        cash=5000.0,
        positions={"btc_ohlcv": 0.1},
        portfolio_value=10000.0,
        daily_pnl=100.0,
        daily_return=0.01,
        combined_signal=0.5,
        dd_scale=1.0,
        vol_scale=1.0,
    )
    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.get_last_snapshot.return_value = snapshot

    trader.executor = MagicMock()
    trader.executor.get_reconciled_position.return_value = 0.15  # Mismatch
    trader.executor.get_reconciled_cash.return_value = 4800.0  # Mismatch

    result = trader.reconcile()
    assert result["match"] is False
    assert result["internal_qty"] == pytest.approx(0.1)
    assert result["qty_diff"] == pytest.approx(0.05)
    assert result["cash_diff"] == pytest.approx(200.0)


def test_reconcile_no_data():
    """reconcile() returns no_data when no snapshots exist."""
    from alpha_os.paper.trader import Trader

    trader = object.__new__(Trader)
    trader.price_signal = "btc_ohlcv"

    trader.portfolio_tracker = MagicMock()
    trader.portfolio_tracker.get_last_snapshot.return_value = None

    result = trader.reconcile()
    assert result["status"] == "no_data"


def test_cmd_runtime_status_shows_registry_and_report(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace

    from alpha_os.alpha.managed_alphas import AlphaRecord, ManagedAlphaStore, AlphaState
    from alpha_os.cli import cmd_runtime_status
    from alpha_os.validation.testnet import readiness_paths

    reg = ManagedAlphaStore(tmp_path / "alpha_registry.db")
    reg.register(AlphaRecord(alpha_id="a1", expression="x", state=AlphaState.ACTIVE))
    reg.register(AlphaRecord(alpha_id="a2", expression="y", state=AlphaState.DORMANT))
    reg.register(AlphaRecord(alpha_id="a3", expression="z", state=AlphaState.REJECTED))
    reg.replace_deployed_alphas(["a1"])
    reg.close()

    state_path, report_path = readiness_paths(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "consecutive_success_days": 3,
        "total_days_run": 9,
        "last_success_date": "2026-03-09",
        "last_run_date": "2026-03-09",
        "last_profile_id": "prof123456789",
        "target_days": 10,
        "passed": False,
    }))
    report_path.write_text(json.dumps({
        "date": "2026-03-09",
        "profile_id": "prof123456789",
        "profile_commit": "deadbeefcafebabe",
        "profile_config_id": "cfg123456789",
        "profile_deployed_set_id": "dep123456789",
        "portfolio_value": 9905.91,
        "daily_pnl": 0.0,
        "n_fills": 0,
        "n_registry_active": 7,
        "n_deployed_alphas": 1,
        "n_selected_alphas": 1,
        "n_skipped_deadband": 1,
        "n_skipped_min_notional": 0,
        "n_skipped_rounded_to_zero": 0,
        "n_order_failures": 0,
        "reconciliation_match": True,
        "circuit_breaker_halted": False,
        "has_errors": False,
    }) + "\n")

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)

    cmd_runtime_status(Namespace(asset="BTC", config=None))
    output = capsys.readouterr().out

    assert "Runtime Status (BTC)" in output
    assert "Readiness: 3/10 days" in output
    assert "Managed:   active=1 dormant=1 rejected=1 deployed=1" in output
    assert "Profile:   current=" in output
    assert "Profile:   latest=prof12345678" in output
    assert "ProfileIDs: config=cfg123456789 deployed=dep123456789" in output
    assert "CurrentIDs: config=" in output
    assert "Latest:    2026-03-09 [OK]" in output
    assert "Skips:     deadband=1 min_notional=0 rounded_to_zero=0" in output
    assert "Observe:   pending" in output
    assert "- latest cycle had zero fills" in output
    assert "- deadband skipped the latest cycle" in output
    assert "- latest report was recorded under a different runtime profile" in output
    assert "- config fingerprint differs between current and latest" in output
    assert "- deployed alpha set fingerprint differs between current and latest" in output
    assert "Note:      managed-alpha DB count differs" in output
