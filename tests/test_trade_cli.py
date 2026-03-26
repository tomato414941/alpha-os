"""Tests for the trade CLI command, Trader rename, and Phase 4 features."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


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
    assert args.non_interactive is False
    assert args.strict is False
    assert args.capital == 10000.0
    assert args.asset == "BTC"
    assert not hasattr(args, "tactical")

    # Real mode
    args = parser.parse_args([
        "trade", "--once", "--real", "--non-interactive", "--strict", "--capital", "500",
    ])
    assert args.real is True
    assert args.non_interactive is True
    assert args.strict is True
    assert args.capital == 500.0


def test_hypothesis_seeder_parser_supports_asset_once_and_bootstrap_flags():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["hypothesis-seeder", "--asset", "ETH", "--once", "--skip-bootstrap"]
    )

    assert args.command == "hypothesis-seeder"
    assert args.asset == "ETH"
    assert args.once is True
    assert args.skip_bootstrap is True
    assert args.include_bootstrap is False

    args = parser.parse_args(
        ["hypothesis-seeder", "--asset", "ETH", "--once", "--include-bootstrap"]
    )

    assert args.include_bootstrap is True
    assert args.skip_bootstrap is False


def test_cmd_hypothesis_seeder_uses_asset_default_bootstrap_policy(monkeypatch):
    from alpha_os.cli import cmd_hypothesis_seeder

    captured = {}

    class _Daemon:
        def __init__(self, *, config, primary_asset, include_bootstrap):
            captured["asset"] = primary_asset
            captured["include_bootstrap"] = include_bootstrap

        def _run_round(self):
            return SimpleNamespace(
                generated_dsl=0,
                inserted_dsl=0,
                skipped_dsl=0,
                inserted_bootstrap=0,
                skipped_bootstrap=0,
            )

        def close(self):
            return None

    monkeypatch.setattr("alpha_os.cli._load_config", lambda _path: object())
    monkeypatch.setattr("alpha_os.cli.logging.basicConfig", lambda **_kwargs: None)
    monkeypatch.setattr("alpha_os.daemon.hypothesis_seeder.HypothesisSeederDaemon", _Daemon)

    cmd_hypothesis_seeder(
        SimpleNamespace(
            config=None,
            asset="ETH",
            once=True,
            include_bootstrap=False,
            skip_bootstrap=False,
        )
    )

    assert captured == {"asset": "ETH", "include_bootstrap": None}


def test_cmd_hypothesis_seeder_can_force_bootstrap(monkeypatch):
    from alpha_os.cli import cmd_hypothesis_seeder

    captured = {}

    class _Daemon:
        def __init__(self, *, config, primary_asset, include_bootstrap):
            captured["asset"] = primary_asset
            captured["include_bootstrap"] = include_bootstrap

        def _run_round(self):
            return SimpleNamespace(
                generated_dsl=0,
                inserted_dsl=0,
                skipped_dsl=0,
                inserted_bootstrap=0,
                skipped_bootstrap=0,
            )

        def close(self):
            return None

    monkeypatch.setattr("alpha_os.cli._load_config", lambda _path: object())
    monkeypatch.setattr("alpha_os.cli.logging.basicConfig", lambda **_kwargs: None)
    monkeypatch.setattr("alpha_os.daemon.hypothesis_seeder.HypothesisSeederDaemon", _Daemon)

    cmd_hypothesis_seeder(
        SimpleNamespace(
            config=None,
            asset="ETH",
            once=True,
            include_bootstrap=True,
            skip_bootstrap=False,
        )
    )

    assert captured == {"asset": "ETH", "include_bootstrap": True}


def test_run_sleeves_once_passes_bootstrap_override(monkeypatch, capsys):
    from alpha_os.cli import cmd_run_sleeves_once

    seed_calls = []

    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.build_template_gap_search_budget",
        lambda *, asset, base_limit, previous_template_gaps=None: SimpleNamespace(
            asset=asset,
            requested_limit=base_limit,
            effective_limit=base_limit,
            missing_template_count=1,
            closed_template_count=0,
            new_template_count=0,
        ),
    )
    monkeypatch.setattr("alpha_os.cli.cmd_score_exploratory_hypotheses", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli.cmd_rebalance_allocation_trust", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli.cmd_trade", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli.cmd_runtime_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli._write_sleeve_compare_snapshot", lambda *_args, **_kwargs: "ignored")

    def _seed(args):
        seed_calls.append((args.asset, args.include_bootstrap, args.skip_bootstrap))

    monkeypatch.setattr("alpha_os.cli.cmd_hypothesis_seeder", _seed)

    cmd_run_sleeves_once(
        SimpleNamespace(
            asset="BTC",
            assets="BTC,ETH",
            config=None,
            score_limit=12,
            bootstrap_assets="BTC",
            refresh_bootstrap_assets=True,
            skip_seed=False,
            skip_score=True,
            skip_rebalance=True,
            skip_trade=True,
            skip_status=True,
        )
    )

    assert seed_calls == [
        ("BTC", True, False),
        ("ETH", False, True),
    ]


def test_run_sleeves_once_skips_reference_sleeve_refresh_by_default(monkeypatch):
    from alpha_os.cli import cmd_run_sleeves_once
    from types import SimpleNamespace

    calls = []

    monkeypatch.setattr(
        "alpha_os.cli.cmd_hypothesis_seeder",
        lambda args: calls.append(("seed", args.asset)),
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_score_exploratory_hypotheses",
        lambda args: calls.append(("score", args.asset)),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.build_template_gap_search_budget",
        lambda *, asset, base_limit, previous_template_gaps=None: SimpleNamespace(
            asset=asset,
            requested_limit=base_limit,
            effective_limit=base_limit,
            missing_template_count=1,
            closed_template_count=0,
            new_template_count=0,
        ),
    )
    monkeypatch.setattr(
        "alpha_os.cli.asset_data_dir",
        lambda asset: __import__("pathlib").Path("/tmp") / asset.lower(),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.store.HypothesisStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.data.store.DataStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.forward.tracker.HypothesisObservationTracker",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_template_service.run_serious_template_maintenance",
        lambda **kwargs: calls.append(("serious", kwargs["asset"]))
        or SimpleNamespace(
            asset=kwargs["asset"],
            template_total=6,
            inserted=0,
            refreshed=0,
            backfill=SimpleNamespace(n_records=0, n_failures=0),
        ),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_templates.serious_seed_specs",
        lambda asset: [object()],
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_rebalance_allocation_trust",
        lambda args: calls.append(("rebalance", args.asset)),
    )
    monkeypatch.setattr("alpha_os.cli.cmd_trade", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli.cmd_runtime_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli._write_sleeve_compare_snapshot", lambda *_args, **_kwargs: "ignored")

    cmd_run_sleeves_once(
        SimpleNamespace(
            asset="BTC",
            assets="BTC,ETH",
            config=None,
            score_limit=12,
            bootstrap_assets="BTC",
            refresh_bootstrap_assets=False,
            skip_seed=False,
            skip_score=False,
            skip_rebalance=False,
            skip_trade=True,
            skip_status=True,
        )
    )

    assert calls == [
        ("seed", "ETH"),
        ("serious", "BTC"),
        ("serious", "ETH"),
        ("score", "ETH"),
        ("rebalance", "BTC"),
        ("rebalance", "ETH"),
    ]


def test_backfill_observation_returns_parser_supports_source_filter():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["backfill-observation-returns", "--asset", "BTC", "--days", "15", "--source", "bootstrap_serious"]
    )

    assert args.command == "backfill-observation-returns"
    assert args.asset == "BTC"
    assert args.days == 15
    assert args.source == "bootstrap_serious"


def test_trade_runtime_lock_path():
    from alpha_os.runtime_lock import runtime_lock_path

    path = runtime_lock_path("trade", ["eth", "btc"])

    assert path.name == "trade_BTC-ETH.lock"
    assert path.parent.name == "locks"


def test_build_signal_activity_getter_uses_positive_activity_for_long_only():
    from alpha_os.cli import _build_signal_activity_getter

    class _Tracker:
        def get_hypothesis_signal_history(self, hypothesis_id, limit=20):
            assert hypothesis_id == "h1"
            assert limit == 5
            return [-1.0, -0.5, 0.0, 0.2, 0.4]

    getter = _build_signal_activity_getter(
        _Tracker(),
        None,
        lookback=5,
        supports_short=False,
    )

    assert getter is not None
    ratio, mean_signal = getter("h1")
    assert ratio == pytest.approx(2 / 5)
    assert mean_signal == pytest.approx((0.2 + 0.4) / 5)


def test_build_signal_activity_getter_uses_abs_activity_for_shortable_runtime():
    from alpha_os.cli import _build_signal_activity_getter

    class _Tracker:
        def get_hypothesis_signal_history(self, hypothesis_id, limit=20):
            assert hypothesis_id == "h1"
            assert limit == 5
            return [-1.0, -0.5, 0.0, 0.2, 0.4]

    getter = _build_signal_activity_getter(
        _Tracker(),
        None,
        lookback=5,
        supports_short=True,
    )

    assert getter is not None
    ratio, mean_signal = getter("h1")
    assert ratio == pytest.approx(4 / 5)
    assert mean_signal == pytest.approx((1.0 + 0.5 + 0.2 + 0.4) / 5)


def test_build_asset_sleeve_summary_tracks_serious_cohort():
    from alpha_os.hypotheses.sleeve_status import build_asset_sleeve_summary
    from alpha_os.hypotheses.store import HypothesisKind, HypothesisRecord

    summary = build_asset_sleeve_summary([
        HypothesisRecord(
            hypothesis_id="serious_retained",
            kind=HypothesisKind.DSL,
            definition={"expression": "btc_difficulty"},
            source="bootstrap_serious",
            stake=0.2,
            metadata={
                "lifecycle_research_retained": True,
                "lifecycle_capital_backed": True,
                "lifecycle_bootstrap_trust": 0.09,
            },
        )
    ])

    assert summary.research_retained == 1
    assert summary.serious_research_retained == 1
    assert summary.serious_capital_backed == 1
    assert summary.bootstrap_research_retained == 0
    assert summary.bootstrap_capital_backed == 0


def test_build_asset_sleeve_summary_counts_live_serious_source():
    from alpha_os.hypotheses.sleeve_status import build_asset_sleeve_summary
    from alpha_os.hypotheses.store import HypothesisKind, HypothesisRecord

    summary = build_asset_sleeve_summary([
        HypothesisRecord(
            hypothesis_id="serious_live",
            kind=HypothesisKind.DSL,
            definition={"expression": "btc_difficulty"},
            source="bootstrap_serious",
            stake=0.3,
            metadata={
                "lifecycle_live_proven": True,
                "lifecycle_actionable_live": True,
                "lifecycle_capital_backed": True,
                "lifecycle_bootstrap_trust": 0.08,
                "serious_template": "macro_dollar_pressure",
            },
        )
    ])

    assert summary.live_proven == 1
    assert summary.serious_research_retained == 1
    assert summary.serious_capital_backed == 1
    assert summary.actionable_live_capital_backed == 0
    assert summary.serious_template_target_count == 9
    assert summary.serious_template_retained_count == 1
    assert summary.serious_template_backed_count == 1
    assert summary.serious_retained_templates == ["macro_dollar_pressure:1"]
    assert summary.serious_backed_templates == ["macro_dollar_pressure:1"]
    assert len(summary.serious_template_gaps) == 3
    assert all(item.endswith(":1.00") for item in summary.serious_template_gaps)


def test_needs_trade_evolution_supports_hypothesis_registry():
    from alpha_os.cli import _needs_trade_evolution

    class _HypothesisRegistry:
        def __init__(self, active_count):
            self._active_count = active_count

        def list_observation_active(self, *, asset=None):
            return [object()] * self._active_count

    trader = SimpleNamespace(registry=_HypothesisRegistry(active_count=1))
    assert _needs_trade_evolution(trader) is False

    empty_trader = SimpleNamespace(registry=_HypothesisRegistry(active_count=0))
    assert _needs_trade_evolution(empty_trader) is True


def test_trade_skips_overlapping_invocation(monkeypatch, capsys):
    from alpha_os.cli import cmd_trade
    from alpha_os.runtime_lock import RuntimeLockBusy

    cfg = SimpleNamespace(
        forward=SimpleNamespace(check_interval=14400),
        regime=SimpleNamespace(enabled=False),
        paper=SimpleNamespace(),
    )
    args = SimpleNamespace(
        config=None,
        interval=None,
        real=False,
        non_interactive=False,
        asset="BTC",
        assets=None,
        capital=10000.0,
        strict=False,
        summary=False,
        once=True,
        schedule=False,
        event_driven=False,
        evolve_interval=86400,
        pop_size=200,
        generations=30,
        debounce=None,
    )

    class BusyLock:
        def __enter__(self):
            raise RuntimeLockBusy(Path("/tmp/trade_BTC.lock"))

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("alpha_os.cli._load_config", lambda _path: cfg)
    monkeypatch.setattr("alpha_os.cli._normalize_trade_config", lambda _cfg: [])
    monkeypatch.setattr("alpha_os.cli._resolve_asset_list", lambda _args: ["BTC"])
    monkeypatch.setattr("alpha_os.cli.hold_runtime_lock", lambda _path: BusyLock())
    monkeypatch.setattr("alpha_os.cli.logging.basicConfig", lambda **_kwargs: None)

    cmd_trade(args)

    out = capsys.readouterr().out
    assert "Trade runtime already active for BTC" in out


def test_trade_strict_exits_on_overlapping_invocation(monkeypatch):
    from alpha_os.cli import cmd_trade
    from alpha_os.runtime_lock import RuntimeLockBusy

    cfg = SimpleNamespace(
        forward=SimpleNamespace(check_interval=14400),
        regime=SimpleNamespace(enabled=False),
        paper=SimpleNamespace(),
    )
    args = SimpleNamespace(
        config=None,
        interval=None,
        real=False,
        non_interactive=False,
        asset="BTC",
        assets=None,
        capital=10000.0,
        strict=True,
        summary=False,
        once=True,
        schedule=False,
        event_driven=False,
        evolve_interval=86400,
        pop_size=200,
        generations=30,
        debounce=None,
        venue=None,
    )

    class BusyLock:
        def __enter__(self):
            raise RuntimeLockBusy(Path("/tmp/trade_BTC.lock"))

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("alpha_os.cli._load_config", lambda _path: cfg)
    monkeypatch.setattr("alpha_os.cli._normalize_trade_config", lambda _cfg: [])
    monkeypatch.setattr("alpha_os.cli._resolve_asset_list", lambda _args: ["BTC"])
    monkeypatch.setattr("alpha_os.cli.hold_runtime_lock", lambda _path: BusyLock())
    monkeypatch.setattr("alpha_os.cli.logging.basicConfig", lambda **_kwargs: None)

    with pytest.raises(SystemExit):
        cmd_trade(args)


def test_run_hypothesis_lifecycle_update_updates_stakes(tmp_path):
    from types import SimpleNamespace

    from alpha_os.cli import _run_hypothesis_lifecycle_update
    from alpha_os.config import Config
    from alpha_os.hypotheses import HypothesisKind, HypothesisRecord, HypothesisStore
    from alpha_os.paper.tracker import PaperPortfolioTracker

    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "x"},
            stake=1.0,
            metadata={"oos_log_growth": 0.8},
        )
    )
    tracker = PaperPortfolioTracker(tmp_path / "paper.db")
    tracker.save_hypothesis_signals("2026-03-23T05:50:27", {"h1": 1.0})

    trader = SimpleNamespace(
        registry=store,
        portfolio_tracker=tracker,
    )
    cfg = Config()
    cfg.portfolio.objective = "log_growth"
    cfg.forward.degradation_window = 3
    cfg.live_quality.min_observations = 1
    cfg.live_quality.full_weight_observations = 1

    result = SimpleNamespace(
        date="2026-03-23T05:50:27",
        daily_return=0.2,
    )

    updates = _run_hypothesis_lifecycle_update(trader, cfg, result)
    updated = store.get("h1")

    assert "h1" in updates
    assert updated is not None
    assert updated.stake != pytest.approx(1.0)

    tracker.close()
    store.close()


def test_trade_once_strict_failure_detects_empty_live_set():
    from alpha_os.cli import _trade_once_strict_failure

    result = SimpleNamespace(
        n_live_hypotheses=0,
        n_signals_evaluated=0,
        order_failures=0,
    )

    assert _trade_once_strict_failure("BTC", result) == "BTC: no live hypotheses"


def test_trade_once_strict_failure_detects_order_failures():
    from alpha_os.cli import _trade_once_strict_failure

    result = SimpleNamespace(
        n_live_hypotheses=5,
        n_signals_evaluated=5,
        order_failures=2,
    )

    assert _trade_once_strict_failure("BTC", result) == "BTC: order failures=2"


def test_trade_once_status_reports_traded():
    from alpha_os.cli import _trade_once_status

    result = SimpleNamespace(
        n_live_hypotheses=5,
        n_signals_evaluated=5,
        order_failures=0,
        fills=[object()],
        n_skipped_deadband=0,
        n_skipped_no_delta=0,
        n_skipped_min_notional=0,
        n_skipped_rounded_to_zero=0,
    )

    assert _trade_once_status(result) == "traded"


def test_compute_stake_weights_enforces_final_max_weight():
    from alpha_os.hypotheses.combiner import compute_stake_weights

    weights = compute_stake_weights(
        {
            "a": 10.0,
            "b": 9.0,
            "c": 8.0,
            "d": 7.0,
            "e": 6.0,
            "f": 5.0,
            "g": 4.0,
            "h": 3.0,
            "i": 2.0,
            "j": 1.0,
            "k": 1.0,
            "l": 1.0,
            "m": 1.0,
            "n": 1.0,
            "o": 1.0,
            "p": 1.0,
            "q": 1.0,
            "r": 1.0,
            "s": 1.0,
            "t": 1.0,
        },
        max_weight=0.10,
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 0.1000001


def test_trade_once_status_reports_no_delta():
    from alpha_os.cli import _trade_once_status

    result = SimpleNamespace(
        n_live_hypotheses=5,
        n_signals_evaluated=5,
        order_failures=0,
        fills=[],
        n_skipped_deadband=0,
        n_skipped_no_delta=1,
        n_skipped_min_notional=0,
        n_skipped_rounded_to_zero=0,
    )

    assert _trade_once_status(result) == "no_delta"


def test_research_paper_replay_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "research", "paper-replay", "--start", "2025-09-01", "--end", "2026-03-05",
    ])

    assert args.command == "research"
    assert args.research_command == "paper-replay"
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
    with pytest.raises(SystemExit):
        parser.parse_args(["cross-trade", "--assets", "BTC,ETH"])


def test_produce_predictions_prefers_hypotheses(monkeypatch, capsys):
    from alpha_os.cli import cmd_produce_predictions

    args = SimpleNamespace(config=None, asset="BTC", strict=False)
    calls = {}

    monkeypatch.setattr("alpha_os.config.Config.load", lambda _path: object())

    def _produce(cfg, *, assets=None, **_kwargs):
        calls["assets"] = assets
        return 7

    monkeypatch.setattr(
        "alpha_os.hypotheses.producer.produce_active_hypothesis_predictions",
        _produce,
    )

    cmd_produce_predictions(args)

    assert calls["assets"] == ["BTC"]
    out = capsys.readouterr().out
    assert "Wrote 7 hypothesis predictions to store" in out
    assert "Prediction summary: asset=BTC status=ok written=7" in out


def test_produce_predictions_no_longer_uses_registry_fallback(monkeypatch, capsys):
    from alpha_os.cli import cmd_produce_predictions

    args = SimpleNamespace(config=None, asset="ETH", strict=False)
    calls = {}

    monkeypatch.setattr("alpha_os.config.Config.load", lambda _path: object())

    def _produce(cfg, *, assets=None, **_kwargs):
        calls["assets"] = assets
        return 0

    monkeypatch.setattr(
        "alpha_os.hypotheses.producer.produce_active_hypothesis_predictions",
        _produce,
    )

    cmd_produce_predictions(args)

    assert calls["assets"] == ["ETH"]
    assert "Wrote 0 hypothesis predictions to store" in capsys.readouterr().out


def test_produce_predictions_strict_exits_on_zero(monkeypatch):
    from alpha_os.cli import cmd_produce_predictions

    args = SimpleNamespace(config=None, asset="ETH", strict=True)

    monkeypatch.setattr("alpha_os.config.Config.load", lambda _path: object())
    monkeypatch.setattr(
        "alpha_os.hypotheses.producer.produce_active_hypothesis_predictions",
        lambda _cfg, *, assets=None, **_kwargs: 0,
    )

    with pytest.raises(SystemExit):
        cmd_produce_predictions(args)


def test_produce_predictions_skips_overlapping_invocation(monkeypatch, capsys):
    from alpha_os.cli import cmd_produce_predictions
    from alpha_os.runtime_lock import RuntimeLockBusy

    args = SimpleNamespace(config=None, asset="BTC", strict=False)

    class BusyLock:
        def __enter__(self):
            raise RuntimeLockBusy(Path("/tmp/produce-predictions_BTC.lock"))

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("alpha_os.cli.hold_runtime_lock", lambda _path: BusyLock())

    cmd_produce_predictions(args)

    assert "Prediction production already active for BTC" in capsys.readouterr().out


def test_produce_predictions_strict_exits_on_overlapping_invocation(monkeypatch):
    from alpha_os.cli import cmd_produce_predictions
    from alpha_os.runtime_lock import RuntimeLockBusy

    args = SimpleNamespace(config=None, asset="BTC", strict=True)

    class BusyLock:
        def __enter__(self):
            raise RuntimeLockBusy(Path("/tmp/produce-predictions_BTC.lock"))

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("alpha_os.cli.hold_runtime_lock", lambda _path: BusyLock())

    with pytest.raises(SystemExit):
        cmd_produce_predictions(args)


def test_runtime_status_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "runtime-status",
        "--asset", "BTC",
    ])

    assert args.command == "runtime-status"
    assert args.asset == "BTC"


def test_analyze_latest_combine_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "analyze-latest-combine",
        "--asset", "BTC",
        "--top", "3",
    ])

    assert args.command == "analyze-latest-combine"
    assert args.asset == "BTC"
    assert args.top == 3


def test_sync_signal_cache_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "sync-signal-cache",
        "--asset", "BTC",
        "--from-hypotheses",
        "--min-history-days", "30",
    ])

    assert args.command == "sync-signal-cache"
    assert args.asset == "BTC"
    assert args.from_hypotheses is True
    assert args.min_history_days == 30
    assert args.strict is False


def test_sync_signal_cache_strict_exits_when_healthcheck_fails(monkeypatch):
    from alpha_os.cli import cmd_sync_signal_cache

    args = SimpleNamespace(
        config=None,
        asset="BTC",
        assets=None,
        signals=None,
        from_hypotheses=False,
        resolution="1d",
        min_history_days=0,
        strict=True,
    )

    monkeypatch.setattr(
        "alpha_os.cli.Config.load",
        lambda _path: SimpleNamespace(api=SimpleNamespace(base_url="https://example.test")),
    )
    monkeypatch.setattr(
        "alpha_os.cli.build_signal_client_from_config",
        lambda _api: object(),
    )
    monkeypatch.setattr(
        "alpha_os.cli._resolve_signal_cache_targets",
        lambda _args: ["btc_ohlcv"],
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.producer._quick_healthcheck",
        lambda _url: False,
    )

    with pytest.raises(SystemExit):
        cmd_sync_signal_cache(args)


def test_sync_signal_cache_reports_summary_on_overlap(monkeypatch, capsys):
    from alpha_os.cli import cmd_sync_signal_cache
    from alpha_os.runtime_lock import RuntimeLockBusy

    args = SimpleNamespace(
        config=None,
        asset="BTC",
        assets=None,
        signals=None,
        from_hypotheses=False,
        resolution="1d",
        min_history_days=0,
        strict=False,
    )

    class BusyLock:
        def __enter__(self):
            raise RuntimeLockBusy(Path("/tmp/sync-signal-cache_BTC.lock"))

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("alpha_os.cli.hold_runtime_lock", lambda _path: BusyLock())

    cmd_sync_signal_cache(args)

    out = capsys.readouterr().out
    assert "Signal cache sync already active for BTC" in out
    assert "Sync summary: assets=BTC status=skipped_overlap" in out


def test_sync_signal_cache_skips_overlapping_invocation(monkeypatch, capsys):
    from alpha_os.cli import cmd_sync_signal_cache
    from alpha_os.runtime_lock import RuntimeLockBusy

    args = SimpleNamespace(
        config=None,
        asset="BTC",
        assets=None,
        signals=None,
        from_hypotheses=False,
        resolution="1d",
        min_history_days=0,
        strict=False,
    )

    class BusyLock:
        def __enter__(self):
            raise RuntimeLockBusy(Path("/tmp/sync-signal-cache_BTC.lock"))

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("alpha_os.cli.hold_runtime_lock", lambda _path: BusyLock())

    cmd_sync_signal_cache(args)

    assert "Signal cache sync already active for BTC" in capsys.readouterr().out


def test_sync_signal_cache_strict_exits_on_overlapping_invocation(monkeypatch):
    from alpha_os.cli import cmd_sync_signal_cache
    from alpha_os.runtime_lock import RuntimeLockBusy

    args = SimpleNamespace(
        config=None,
        asset="BTC",
        assets=None,
        signals=None,
        from_hypotheses=False,
        resolution="1d",
        min_history_days=0,
        strict=True,
    )

    class BusyLock:
        def __enter__(self):
            raise RuntimeLockBusy(Path("/tmp/sync-signal-cache_BTC.lock"))

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("alpha_os.cli.hold_runtime_lock", lambda _path: BusyLock())

    with pytest.raises(SystemExit):
        cmd_sync_signal_cache(args)


def test_resolve_signal_cache_targets_defaults_to_price_signal():
    from alpha_os.cli import _resolve_signal_cache_targets

    args = SimpleNamespace(asset="BTC", assets=None, signals=None, from_hypotheses=False)

    assert _resolve_signal_cache_targets(args) == ["btc_ohlcv"]


def test_resolve_signal_cache_targets_can_include_hypothesis_features(tmp_path, monkeypatch):
    from alpha_os.cli import _resolve_signal_cache_targets
    from alpha_os.hypotheses import HypothesisKind, HypothesisRecord, HypothesisStore

    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(
        HypothesisRecord(
            hypothesis_id="h1",
            kind=HypothesisKind.DSL,
            definition={"expression": "(sub fear_greed dxy)"},
            stake=1.0,
        )
    )
    store.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", tmp_path / "hypotheses.db")
    args = SimpleNamespace(asset="BTC", assets=None, signals=None, from_hypotheses=True)

    targets = _resolve_signal_cache_targets(args)

    assert "btc_ohlcv" in targets
    assert "fear_greed" in targets
    assert "dxy" in targets


def test_legacy_alpha_funnel_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "legacy",
        "alpha-funnel",
        "--asset", "BTC",
    ])

    assert args.command == "legacy"
    assert args.legacy_command == "alpha-funnel"
    assert args.asset == "BTC"


def test_legacy_enqueue_discovery_pool_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "legacy",
        "enqueue-discovery-pool",
        "--asset", "BTC",
        "--limit", "12",
        "--dry-run",
    ])

    assert args.command == "legacy"
    assert args.legacy_command == "enqueue-discovery-pool"
    assert args.asset == "BTC"
    assert args.limit == 12
    assert args.dry_run is True


def test_legacy_prune_stale_candidates_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "legacy",
        "prune-stale-candidates",
        "--asset", "BTC",
        "--max-age-days", "14",
        "--dry-run",
    ])

    assert args.command == "legacy"
    assert args.legacy_command == "prune-stale-candidates"
    assert args.asset == "BTC"
    assert args.max_age_days == 14
    assert args.dry_run is True


def test_legacy_lifecycle_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "legacy",
        "lifecycle",
        "--asset", "ETH",
        "--config", "prod.toml",
    ])

    assert args.command == "legacy"
    assert args.legacy_command == "lifecycle"
    assert args.asset == "ETH"
    assert args.config == "prod.toml"


@pytest.mark.parametrize(
    "command",
    [
        "rebuild-managed-alphas",
        "refresh-deployed-alphas",
        "prune-managed-alpha-duplicates",
        "seed-handcrafted",
        "analyze-diversity",
        "submit",
        "admission-daemon",
        "prune-stale-candidates",
        "replay-experiment",
        "replay-matrix",
        "paper",
        "generate",
        "backtest",
        "evolve",
        "validate",
        "evaluate",
        "produce-classical",
    ],
)
def test_legacy_registry_commands_are_not_parsed(command):
    from alpha_os.cli import _build_parser

    parser = _build_parser()

    argv = [command]
    if command == "paper":
        argv = ["paper", "--replay", "--start", "2025-09-01", "--end", "2026-03-05"]

    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_research_replay_experiment_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "research",
        "replay-experiment",
        "--name", "confidence sweep",
        "--start", "2026-02-20",
        "--end", "2026-03-05",
        "--managed-alpha-mode", "admission",
        "--source", "candidates",
        "--deployment-mode", "refresh",
        "--set", "lifecycle.candidate_quality_min=1.10",
        "--set", "live_quality.min_observations=30",
    ])

    assert args.command == "research"
    assert args.research_command == "replay-experiment"
    assert args.name == "confidence sweep"
    assert args.managed_alpha_mode == "admission"
    assert args.source == "candidates"
    assert args.deployment_mode == "refresh"
    assert args.set == [
        "lifecycle.candidate_quality_min=1.10",
        "live_quality.min_observations=30",
    ]


def test_research_replay_matrix_parser():
    from alpha_os.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args([
        "research",
        "replay-matrix",
        "--manifest", "experiments/deadband.toml",
        "--max-workers", "4",
    ])

    assert args.command == "research"
    assert args.research_command == "replay-matrix"
    assert args.manifest == "experiments/deadband.toml"
    assert args.max_workers == 4


def test_cmd_replay_experiment_prints_profile(capsys, monkeypatch):
    from argparse import Namespace

    from alpha_os.cli import cmd_replay_experiment
    from alpha_os.legacy.replay_experiment import ReplayExperimentRun

    monkeypatch.setattr(
        "alpha_os.legacy.replay_experiment.run_replay_experiment",
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
    from alpha_os.legacy.replay_experiment import ReplayExperimentRun, ReplayExperimentSpec

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
    cfg.regime.enabled = True

    changes = _normalize_trade_config(cfg)

    assert cfg.regime.enabled is True
    assert changes == []


def test_load_runtime_observation_config_prefers_user_prod(tmp_path, monkeypatch):
    from alpha_os.cli import _load_runtime_observation_config

    home = tmp_path / "home"
    prod = home / ".config" / "alpha-os" / "prod.toml"
    prod.parent.mkdir(parents=True, exist_ok=True)
    prod.write_text("[deployment]\nmax_deployed_alphas = 150\n")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)

    cfg = _load_runtime_observation_config(None)

    assert cfg.deployment.max_deployed_alphas == 150


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
        n_active_hypotheses=615,
        n_live_hypotheses=150,
        n_shortlist_candidates=150,
        n_selected_hypotheses=30,
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
    assert "Active:     615 hypotheses" in output
    assert "Live:       150 hypotheses" in output
    assert "Shortlist:  150 candidates" in output
    assert "Selected:   30 hypotheses" in output
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
        n_active_hypotheses=615,
        n_live_hypotheses=30,
        n_shortlist_candidates=30,
        n_selected_hypotheses=30,
        n_signals_evaluated=30,
        n_skipped_deadband=1,
        n_skipped_no_delta=2,
        n_skipped_min_notional=2,
        n_skipped_rounded_to_zero=3,
    )

    _print_paper_result(result)
    output = capsys.readouterr().out

    assert "Skips:      deadband=1" in output
    assert "Skips:      no_delta=2" in output
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


def test_cmd_runtime_status_shows_hypotheses_and_report(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace

    from alpha_os.cli import cmd_runtime_status
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStatus, HypothesisStore
    from alpha_os.validation.testnet import readiness_paths

    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(HypothesisRecord(
        hypothesis_id="h1",
        kind="dsl",
        definition={"expression": "x"},
        stake=1.0,
        source="random_dsl",
        metadata={
            "lifecycle_live_quality": 0.3,
            "lifecycle_raw_live_quality": 12.0,
            "lifecycle_bootstrap_trust": 0.0,
            "lifecycle_quality_confidence": 0.5,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="h4",
        kind="technical",
        definition={"indicator": "rsi"},
        stake=0.5,
        source="bootstrap_technical",
        metadata={
            "lifecycle_live_quality": 0.2,
            "lifecycle_raw_live_quality": 8.0,
            "lifecycle_bootstrap_trust": 0.05,
            "lifecycle_quality_confidence": 0.25,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="h2",
        kind="dsl",
        definition={"expression": "y"},
        status=HypothesisStatus.PAUSED,
        stake=1.0,
    ))
    store.register(HypothesisRecord(
        hypothesis_id="h3",
        kind="dsl",
        definition={"expression": "z"},
        status=HypothesisStatus.ARCHIVED,
        stake=0.0,
    ))
    store.close()

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
        "profile_live_set_id": "live123456789",
        "portfolio_value": 9905.91,
        "daily_pnl": 0.0,
        "n_fills": 0,
        "n_active_hypotheses": 7,
        "n_live_hypotheses": 1,
        "n_selected_hypotheses": 1,
        "n_skipped_deadband": 1,
        "n_skipped_no_delta": 2,
        "n_skipped_min_notional": 0,
        "n_skipped_rounded_to_zero": 0,
        "n_order_failures": 0,
        "reconciliation_match": True,
        "circuit_breaker_halted": False,
        "has_errors": False,
    }) + "\n")

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", tmp_path / "hypotheses.db")

    cmd_runtime_status(Namespace(asset="BTC", config=None))
    output = capsys.readouterr().out

    assert "Runtime Status (BTC)" in output
    assert "Readiness: 3/10 days" in output
    assert "Runtime:   source=hypotheses live=2" in output
    assert "Hypotheses: active=2 paused=1 archived=1 live=2" in output
    assert (
        "Signals:   observed=2 bootstrap_backed=1 research_retained=0 "
        "live_proven=0 actionable_live=0 promoted_live=0 research_demoted=0 "
        "research_candidate_capped=0 capital_backed=2"
    ) in output
    assert "Cohorts:   bootstrap=0/2 serious=0/0 batch=0/0 live=0/0" in output
    assert "TopAlloc:  h1=1.000(dsl/random_dsl), h4=0.500(technical/bootstrap_technical)" in output
    assert "TopEff:    h1=0.300(dsl/random_dsl), h4=0.200(technical/bootstrap_technical)" in output
    assert "TopRaw:    h1=12.000(dsl/random_dsl), h4=8.000(technical/bootstrap_technical)" in output
    assert "TopBoot:   h4=0.050(technical/bootstrap_technical), h1=0.000(dsl/random_dsl)" in output
    assert "Profile:   current=" in output
    assert "Profile:   latest=prof12345678" in output
    assert "ProfileIDs: config=cfg123456789 live=live12345678" in output
    assert "CurrentIDs: config=" in output
    assert "Latest:    2026-03-09 [OK]" in output
    assert "Skips:     deadband=1 no_delta=2 min_notional=0 rounded_to_zero=0" in output
    assert "Observe:   watch" in output
    assert "- latest cycle had zero fills" in output
    assert "- latest cycle had no_delta in long-only mode" in output
    assert "- deadband skipped the latest cycle" in output
    assert "- latest report was recorded under a different runtime profile" in output
    assert "- config fingerprint differs between current and latest" in output
    assert "- live hypothesis set fingerprint differs between current and latest" in output
    assert "Note:      current runtime live count differs" in output


def test_cmd_runtime_status_shows_ok_observation_when_no_findings(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace

    from alpha_os.cli import cmd_runtime_status
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore
    from alpha_os.validation.testnet import readiness_paths

    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(HypothesisRecord(
        hypothesis_id="h1",
        kind="dsl",
        definition={"expression": "x"},
        stake=1.0,
    ))
    store.close()

    state_path, report_path = readiness_paths(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "consecutive_success_days": 2,
        "total_days_run": 2,
        "last_success_date": "2026-03-23",
        "last_run_date": "2026-03-23",
        "last_profile_id": "",
        "target_days": 10,
        "passed": False,
    }))
    report_path.write_text(json.dumps({
        "date": "2026-03-23",
        "profile_id": "",
        "profile_commit": "",
        "profile_config_id": "",
        "profile_live_set_id": "",
        "portfolio_value": 9986.38,
        "daily_pnl": 279.60,
        "n_fills": 1,
        "n_active_hypotheses": 1,
        "n_live_hypotheses": 1,
        "n_selected_hypotheses": 1,
        "n_skipped_deadband": 0,
        "n_skipped_no_delta": 0,
        "n_skipped_min_notional": 0,
        "n_skipped_rounded_to_zero": 0,
        "n_order_failures": 0,
        "reconciliation_match": True,
        "circuit_breaker_halted": False,
        "has_errors": False,
    }) + "\n")

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", tmp_path / "hypotheses.db")

    cmd_runtime_status(Namespace(asset="BTC", config=None))
    output = capsys.readouterr().out

    assert "Observe:   ok" in output


def test_cmd_runtime_status_surfaces_live_promotion_blockers(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace

    from alpha_os.cli import cmd_runtime_status
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore
    from alpha_os.validation.testnet import readiness_paths

    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(HypothesisRecord(
        hypothesis_id="h1",
        kind="dsl",
        definition={"expression": "x"},
        stake=0.0,
        metadata={
            "lifecycle_live_promotion_blocker": "insufficient_observations",
            "lifecycle_quality_confidence": 0.0,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="h2",
        kind="dsl",
        definition={"expression": "y"},
        stake=0.0,
        metadata={
            "lifecycle_live_promotion_blocker": "weak_live_quality",
            "lifecycle_quality_confidence": 0.5,
        },
    ))
    store.close()

    state_path, report_path = readiness_paths(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "consecutive_success_days": 0,
        "total_days_run": 0,
        "last_success_date": None,
        "last_run_date": None,
        "last_profile_id": "",
        "target_days": 10,
        "passed": False,
    }))
    report_path.write_text("")

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", tmp_path / "hypotheses.db")

    cmd_runtime_status(Namespace(asset="BTC", config=None))
    output = capsys.readouterr().out

    assert "Cohorts:   bootstrap=0/0 serious=0/0 batch=0/0 live=0/0" in output
    assert "Promote:   obs=1 quality=1 contrib=0 both=0 signal=0" in output


def test_cmd_runtime_status_shows_actionable_drop_breakdown(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace

    from alpha_os.cli import cmd_runtime_status
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore
    from alpha_os.validation.testnet import readiness_paths

    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(HypothesisRecord(
        hypothesis_id="h1",
        kind="dsl",
        definition={"expression": "x"},
        stake=1.0,
        metadata={"lifecycle_actionable_live": True, "lifecycle_live_proven": True},
    ))
    store.register(HypothesisRecord(
        hypothesis_id="h2",
        kind="dsl",
        definition={"expression": "y"},
        stake=0.0,
        metadata={
            "lifecycle_actionable_live": True,
            "lifecycle_live_proven": True,
            "lifecycle_live_quality": 0.4,
            "lifecycle_redundancy_capped_by": "h1",
            "lifecycle_redundancy_correlation": 0.91,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="h3",
        kind="dsl",
        definition={"expression": "z"},
        stake=0.0,
        metadata={"lifecycle_actionable_live": True, "lifecycle_live_proven": True},
    ))
    store.close()

    state_path, report_path = readiness_paths(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "consecutive_success_days": 0,
        "total_days_run": 0,
        "last_success_date": None,
        "last_run_date": None,
        "last_profile_id": "",
        "target_days": 10,
        "passed": False,
    }))
    report_path.write_text("")

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", tmp_path / "hypotheses.db")

    cmd_runtime_status(Namespace(asset="BTC", config=None))
    output = capsys.readouterr().out

    assert "Actionable: backed=1 redundancy_capped=1 other_dropped=1" in output
    assert "TopCap:    h2->h1(corr=0.91)" in output


def test_cmd_runtime_status_shows_batch_family_summary(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace

    from alpha_os.cli import cmd_runtime_status
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore
    from alpha_os.validation.testnet import readiness_paths

    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(HypothesisRecord(
        hypothesis_id="b1",
        kind="dsl",
        definition={"expression": "funding_rate_eth"},
        stake=1.0,
        source="random_dsl",
        metadata={
            "lifecycle_research_retained": True,
            "lifecycle_research_quality_source": "batch_research_score",
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="b2",
        kind="dsl",
        definition={"expression": "oi_eth"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "lifecycle_research_retained": True,
            "lifecycle_research_quality_source": "batch_research_score",
        },
    ))
    store.close()

    state_path, report_path = readiness_paths(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "consecutive_success_days": 0,
        "total_days_run": 0,
        "last_success_date": None,
        "last_run_date": None,
        "last_profile_id": "",
        "target_days": 10,
        "passed": False,
    }))
    report_path.write_text("")

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", tmp_path / "hypotheses.db")

    cmd_runtime_status(Namespace(asset="BTC", config=None))
    output = capsys.readouterr().out

    assert "BatchFam:" in output
    assert "retained=derivatives:2" in output
    assert "backed=derivatives:1" in output


def test_cmd_runtime_status_shows_actionable_window_summary(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace

    from alpha_os.cli import cmd_runtime_status
    from alpha_os.forward.tracker import HypothesisObservationTracker
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore
    from alpha_os.validation.testnet import readiness_paths

    store = HypothesisStore(tmp_path / "hypotheses.db")
    store.register(HypothesisRecord(
        hypothesis_id="h1",
        kind="dsl",
        definition={"expression": "x"},
        stake=1.0,
    ))
    store.register(HypothesisRecord(
        hypothesis_id="h2",
        kind="dsl",
        definition={"expression": "y"},
        stake=1.0,
    ))
    store.close()

    tracker = HypothesisObservationTracker(tmp_path / "forward_returns.db")
    for idx, (h1, h2) in enumerate([(0.3, 0.0), (0.0, 0.2), (0.1, 0.0)], start=1):
        date = f"2026-03-0{idx}"
        tracker.record("h1", date, h1, 0.0)
        tracker.record("h2", date, h2, 0.0)
    tracker.close()

    state_path, report_path = readiness_paths(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "consecutive_success_days": 0,
        "total_days_run": 0,
        "last_success_date": None,
        "last_run_date": None,
        "last_profile_id": "",
        "target_days": 10,
        "passed": False,
    }))
    report_path.write_text("")

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", tmp_path / "hypotheses.db")

    cmd_runtime_status(Namespace(asset="BTC", config=None))
    output = capsys.readouterr().out

    assert "ActionWin: lookback=63 tracked=2 expressing=2" in output


def test_cmd_run_sleeves_once_orchestrates_stages(monkeypatch, capsys):
    from argparse import Namespace
    from types import SimpleNamespace

    from alpha_os.cli import cmd_run_sleeves_once

    calls: list[tuple[str, str, object]] = []

    monkeypatch.setattr(
        "alpha_os.cli.cmd_hypothesis_seeder",
        lambda args: calls.append(("seed", args.asset, args.skip_bootstrap)),
    )
    monkeypatch.setattr(
        "alpha_os.cli.asset_data_dir",
        lambda asset: __import__("pathlib").Path("/tmp") / asset.lower(),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.store.HypothesisStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.data.store.DataStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.forward.tracker.HypothesisObservationTracker",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_template_service.run_serious_template_maintenance",
        lambda **kwargs: calls.append(("serious", kwargs["asset"], kwargs["lookback_days"]))
        or SimpleNamespace(
            asset=kwargs["asset"],
            template_total=6,
            inserted=0,
            refreshed=1,
            backfill=SimpleNamespace(n_records=180, n_failures=0),
        ),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_templates.serious_seed_specs",
        lambda asset: [object()] if asset == "ETH" else [],
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_score_exploratory_hypotheses",
        lambda args: calls.append(("score", args.asset, args.limit)),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.build_template_gap_search_budget",
        lambda *, asset, base_limit, previous_template_gaps=None: SimpleNamespace(
            asset=asset,
            requested_limit=base_limit,
            effective_limit=base_limit,
            missing_template_count=1,
            closed_template_count=0,
            new_template_count=0,
        ),
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_rebalance_allocation_trust",
        lambda args: calls.append(("rebalance", args.asset, args.dry_run)),
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_trade",
        lambda args: calls.append(("trade", args.assets, args.venue)),
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_runtime_status",
        lambda args: calls.append(("status", args.asset, None)),
    )
    monkeypatch.setattr(
        "alpha_os.cli._write_sleeve_compare_snapshot",
        lambda asset_list, config_path, budget_by_asset=None: __import__("pathlib").Path("/tmp/sleeves.jsonl"),
    )

    cmd_run_sleeves_once(
        Namespace(
            asset="BTC",
            assets="BTC,ETH",
            config=None,
            score_limit=12,
            bootstrap_assets="BTC",
            refresh_bootstrap_assets=True,
            skip_seed=False,
            skip_serious=False,
            skip_score=False,
            skip_rebalance=False,
            skip_trade=False,
            skip_status=False,
        )
    )

    assert calls == [
        ("seed", "BTC", False),
        ("seed", "ETH", True),
        ("serious", "ETH", 30),
        ("score", "BTC", 12),
        ("score", "ETH", 12),
        ("rebalance", "BTC", False),
        ("rebalance", "ETH", False),
        ("trade", "BTC,ETH", "paper"),
        ("status", "BTC", None),
        ("status", "ETH", None),
    ]

    output = capsys.readouterr().out
    assert "Sleeve loop [ONCE]: assets=BTC,ETH" in output
    assert "Search budget: requested=12 effective=12 missing=1 closed=0 new=0" in output
    assert "Serious maintenance [APPLY]: asset=ETH" in output
    assert "Sleeve snapshot: /tmp/sleeves.jsonl" in output


def test_cmd_run_sleeves_once_runs_serious_even_when_seed_is_skipped(monkeypatch, capsys):
    from argparse import Namespace
    from types import SimpleNamespace

    from alpha_os.cli import cmd_run_sleeves_once

    calls: list[tuple[str, str, object]] = []

    monkeypatch.setattr(
        "alpha_os.cli.asset_data_dir",
        lambda asset: __import__("pathlib").Path("/tmp") / asset.lower(),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.store.HypothesisStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.data.store.DataStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.forward.tracker.HypothesisObservationTracker",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_template_service.run_serious_template_maintenance",
        lambda **kwargs: calls.append(("serious", kwargs["asset"], kwargs["lookback_days"]))
        or SimpleNamespace(
            asset=kwargs["asset"],
            template_total=6,
            inserted=0,
            refreshed=0,
            backfill=SimpleNamespace(n_records=180, n_failures=0),
        ),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_templates.serious_seed_specs",
        lambda asset: [object()] if asset == "BTC" else [],
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.build_template_gap_search_budget",
        lambda *, asset, base_limit, previous_template_gaps=None: SimpleNamespace(
            asset=asset,
            requested_limit=base_limit,
            effective_limit=base_limit,
            missing_template_count=1,
            closed_template_count=0,
            new_template_count=0,
        ),
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_rebalance_allocation_trust",
        lambda args: calls.append(("rebalance", args.asset, args.dry_run)),
    )
    monkeypatch.setattr(
        "alpha_os.cli._write_sleeve_compare_snapshot",
        lambda asset_list, config_path, budget_by_asset=None: __import__("pathlib").Path("/tmp/sleeves.jsonl"),
    )

    cmd_run_sleeves_once(
        Namespace(
            asset="BTC",
            assets="BTC",
            config=None,
            score_limit=6,
            bootstrap_assets="BTC",
            refresh_bootstrap_assets=False,
            skip_seed=True,
            skip_serious=False,
            skip_score=True,
            skip_rebalance=False,
            skip_trade=True,
            skip_status=True,
        )
    )

    assert calls == [
        ("serious", "BTC", 30),
        ("rebalance", "BTC", False),
    ]
    assert "stages=serious,rebalance" in capsys.readouterr().out


def test_run_sleeves_once_skips_seed_and_score_when_template_gaps_are_closed(monkeypatch):
    from alpha_os.cli import cmd_run_sleeves_once
    from types import SimpleNamespace

    calls = []

    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.build_template_gap_search_budget",
        lambda *, asset, base_limit, previous_template_gaps=None: SimpleNamespace(
            asset=asset,
            requested_limit=base_limit,
            effective_limit=0 if asset == "ETH" else base_limit,
            missing_template_count=0 if asset == "ETH" else 1,
            closed_template_count=0,
            new_template_count=0,
        ),
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_hypothesis_seeder",
        lambda args: calls.append(("seed", args.asset)),
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_score_exploratory_hypotheses",
        lambda args: calls.append(("score", args.asset, args.limit)),
    )
    monkeypatch.setattr(
        "alpha_os.cli.asset_data_dir",
        lambda asset: __import__("pathlib").Path("/tmp") / asset.lower(),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.store.HypothesisStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.data.store.DataStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.forward.tracker.HypothesisObservationTracker",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_template_service.run_serious_template_maintenance",
        lambda **kwargs: SimpleNamespace(
            asset=kwargs["asset"],
            template_total=6,
            inserted=0,
            refreshed=0,
            backfill=SimpleNamespace(n_records=0, n_failures=0),
        ),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_templates.serious_seed_specs",
        lambda asset: [object()],
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_rebalance_allocation_trust",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("alpha_os.cli.cmd_trade", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli.cmd_runtime_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli._write_sleeve_compare_snapshot", lambda *_args, **_kwargs: "ignored")

    cmd_run_sleeves_once(
        SimpleNamespace(
            asset="ETH",
            assets="ETH",
            config=None,
            score_limit=12,
            bootstrap_assets="BTC",
            refresh_bootstrap_assets=False,
            skip_seed=False,
            skip_serious=False,
            skip_score=False,
            skip_rebalance=False,
            skip_trade=True,
            skip_status=True,
        )
    )

    assert calls == []


def test_run_sleeves_once_uses_gap_driven_score_limit(monkeypatch):
    from alpha_os.cli import cmd_run_sleeves_once
    from types import SimpleNamespace

    calls = []

    monkeypatch.setattr(
        "alpha_os.hypotheses.search_budget_service.build_template_gap_search_budget",
        lambda *, asset, base_limit, previous_template_gaps=None: SimpleNamespace(
            asset=asset,
            requested_limit=base_limit,
            effective_limit=6 if asset == "ETH" else base_limit,
            missing_template_count=3 if asset == "ETH" else 1,
            closed_template_count=0,
            new_template_count=0,
        ),
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_hypothesis_seeder",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_score_exploratory_hypotheses",
        lambda args: calls.append((args.asset, args.limit)),
    )
    monkeypatch.setattr(
        "alpha_os.cli.asset_data_dir",
        lambda asset: __import__("pathlib").Path("/tmp") / asset.lower(),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.store.HypothesisStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.data.store.DataStore",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.forward.tracker.HypothesisObservationTracker",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_template_service.run_serious_template_maintenance",
        lambda **kwargs: SimpleNamespace(
            asset=kwargs["asset"],
            template_total=6,
            inserted=0,
            refreshed=0,
            backfill=SimpleNamespace(n_records=0, n_failures=0),
        ),
    )
    monkeypatch.setattr(
        "alpha_os.hypotheses.serious_templates.serious_seed_specs",
        lambda asset: [object()],
    )
    monkeypatch.setattr(
        "alpha_os.cli.cmd_rebalance_allocation_trust",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("alpha_os.cli.cmd_trade", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli.cmd_runtime_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("alpha_os.cli._write_sleeve_compare_snapshot", lambda *_args, **_kwargs: "ignored")
    monkeypatch.setattr("alpha_os.cli._load_previous_sleeve_compare_rows", lambda: {})

    cmd_run_sleeves_once(
        SimpleNamespace(
            asset="ETH",
            assets="ETH",
            config=None,
            score_limit=12,
            bootstrap_assets="BTC",
            refresh_bootstrap_assets=False,
            skip_seed=False,
            skip_serious=False,
            skip_score=False,
            skip_rebalance=False,
            skip_trade=True,
            skip_status=True,
        )
    )

    assert calls == [("ETH", 6)]


def test_cmd_compare_sleeves_reports_key_metrics(monkeypatch, capsys):
    from argparse import Namespace
    from types import SimpleNamespace

    from alpha_os.cli import cmd_compare_sleeves

    monkeypatch.setattr(
        "alpha_os.cli._load_runtime_observation_config",
        lambda _config: SimpleNamespace(
            testnet=SimpleNamespace(target_success_days=10, max_acceptable_slippage_bps=10.0),
            forward=SimpleNamespace(degradation_window=63),
            trading=SimpleNamespace(supports_short=False),
        ),
    )
    monkeypatch.setattr(
        "alpha_os.cli.asset_data_dir",
        lambda asset: __import__("pathlib").Path(f"/tmp/{asset.lower()}"),
    )
    monkeypatch.setattr(
        "alpha_os.validation.testnet.readiness_paths",
        lambda adir: (adir / "state.json", adir / "report.json"),
    )

    latest_by_asset = {
        "BTC": {"has_errors": False, "n_fills": 1},
        "ETH": {"has_errors": False, "n_fills": 0},
    }
    monkeypatch.setattr(
        "alpha_os.cli._load_latest_report",
        lambda report_path: latest_by_asset[report_path.parent.name.upper()],
    )
    monkeypatch.setattr(
        "alpha_os.cli._live_hypothesis_ids",
        lambda asset=None: ["a"] * {"BTC": 20, "ETH": 16}[asset],
    )
    monkeypatch.setattr(
        "alpha_os.cli._runtime_hypothesis_summary",
        lambda asset=None: {
            "BTC": {
                "live_proven": 12,
                "actionable_live": 12,
                "capital_backed": 20,
                "serious_research_retained": 0,
                "serious_capital_backed": 0,
                "serious_template_backed_count": 0,
                "serious_template_target_count": 9,
                "serious_template_gaps": ["onchain_activity_acceleration:1.00"],
            },
            "ETH": {
                "live_proven": 58,
                "actionable_live": 44,
                "capital_backed": 16,
                "serious_research_retained": 0,
                "serious_capital_backed": 0,
                "serious_template_backed_count": 2,
                "serious_template_target_count": 6,
                "serious_template_gaps": ["derivatives_open_interest_trend:1.00"],
            },
        }[asset],
    )
    monkeypatch.setattr(
        "alpha_os.cli._runtime_actionable_window_summary",
        lambda asset=None, lookback=None, supports_short=None: {
            "BTC": {"breadth": 1.00},
            "ETH": {"breadth": 7.46},
        }[asset],
    )
    monkeypatch.setattr(
        "alpha_os.cli._runtime_observation_findings",
        lambda latest, current_live_count: ([] if latest.get("n_fills", 0) > 0 else ["zero_fills"]),
    )
    monkeypatch.setattr(
        "alpha_os.cli._runtime_observation_verdict",
        lambda latest, findings: ("ok" if not findings else "watch"),
    )

    class _FakeReadinessChecker:
        def __init__(self, *, state_path, report_path, target_days, max_slippage_bps):
            asset = state_path.parent.name.upper()
            self.state = {
                "BTC": SimpleNamespace(consecutive_success_days=5, target_days=10),
                "ETH": SimpleNamespace(consecutive_success_days=1, target_days=10),
            }[asset]

    monkeypatch.setattr("alpha_os.validation.testnet.ReadinessChecker", _FakeReadinessChecker)
    snapshot_path = Path("/tmp/test-sleeve-compare.jsonl")
    snapshot_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-25T00:00:00+00:00",
                "assets": ["BTC", "ETH"],
                "rows": [
                    {
                        "asset": "BTC",
                        "serious_template_gaps": [
                            "onchain_activity_acceleration:1.00",
                            "derivatives_open_interest_trend:1.00",
                        ],
                        "backed": 18,
                        "serious_template_backed": 1,
                        "breadth": 0.75,
                        "score_budget_requested": 12,
                        "score_budget_effective": 9,
                    },
                    {
                        "asset": "ETH",
                        "serious_template_gaps": [
                            "derivatives_open_interest_trend:1.00",
                        ],
                        "backed": 15,
                        "serious_template_backed": 1,
                        "breadth": 6.50,
                        "score_budget_requested": 12,
                        "score_budget_effective": 6,
                    },
                ],
            }
        ) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("alpha_os.cli._sleeve_compare_snapshot_path", lambda: snapshot_path)

    cmd_compare_sleeves(Namespace(asset="BTC", assets="BTC,ETH", config=None))
    output = capsys.readouterr().out

    assert "Sleeve Compare: BTC,ETH" in output
    assert "BTC: readiness=5/10 live=20 proven=12 actionable=12 backed=20 serious=0/0 templates=0/9 tpl_gaps=onchain_activity_acceleration:1.00 tpl_delta=closed:1,new:0 budget=9/12 delta=backed:+2,tpl:-1,breadth:+0.25 breadth=1.00 latest=OK fills=1 observe=ok" in output
    assert "ETH: readiness=1/10 live=16 proven=58 actionable=44 backed=16 serious=0/0 templates=2/6 tpl_gaps=derivatives_open_interest_trend:1.00 tpl_delta=closed:0,new:0 budget=6/12 delta=backed:+1,tpl:+1,breadth:+0.96 breadth=7.46 latest=OK fills=0 observe=watch" in output


def test_cmd_analyze_batch_research_shows_drop_reasons(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_batch_research
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="backed",
        kind="dsl",
        definition={"expression": "funding_rate_eth"},
        stake=1.0,
        source="random_dsl",
        metadata={
            "oos_sharpe": 1.2,
            "research_quality_source": "batch_research_score",
            "research_quality_status": "scored",
            "lifecycle_research_backed": True,
            "lifecycle_capital_eligible": True,
            "lifecycle_research_retained": True,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="quality",
        kind="dsl",
        definition={"expression": "close"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "oos_sharpe": 0.1,
            "research_quality_source": "batch_research_score",
            "research_quality_status": "scored",
            "lifecycle_research_backed": False,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="obs",
        kind="dsl",
        definition={"expression": "funding_rate_eth"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "oos_sharpe": 0.8,
            "research_quality_source": "batch_research_score",
            "research_quality_status": "scored",
            "lifecycle_research_backed": True,
            "lifecycle_live_promotion_blocker": "insufficient_observations",
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="signal",
        kind="dsl",
        definition={"expression": "oi_eth"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "oos_sharpe": 0.9,
            "research_quality_source": "batch_research_score",
            "research_quality_status": "scored",
            "lifecycle_research_backed": True,
            "lifecycle_live_promotion_blocker": "weak_signal_activity",
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="redundancy",
        kind="dsl",
        definition={"expression": "funding_rate_eth"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "oos_sharpe": 1.0,
            "research_quality_source": "batch_research_score",
            "research_quality_status": "scored",
            "lifecycle_research_backed": True,
            "lifecycle_redundancy_capped_by": "backed",
        },
    ))
    store.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_batch_research(Namespace(asset="BTC", config=None, top=3))
    output = capsys.readouterr().out

    assert "Batch Research (BTC)" in output
    assert "Summary:  scored=5 retained=1 actionable=0 backed=1" in output
    assert "Drop:     research_q=1 live_q=0 obs=1 signal=1" in output
    assert "redundancy=1" in output
    assert "backed=1" in output
    assert "Fam:" in output
    assert "Quality:  min=0.10" in output
    assert "Inputs:   drop_sharpe_p50=" in output
    assert "FamilyQ:" in output
    assert "Top:      backed reason=backed" in output


def test_cmd_analyze_batch_research_filters_families(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_batch_research
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="onchain_backed",
        kind="dsl",
        definition={"expression": "btc_difficulty"},
        stake=0.2,
        source="random_dsl",
        metadata={
            "oos_sharpe": 0.9,
            "research_quality_source": "batch_research_score",
            "research_quality_status": "scored",
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="price_dropped",
        kind="dsl",
        definition={"expression": "fear_greed"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "oos_sharpe": 0.0,
            "research_quality_source": "batch_research_score",
            "research_quality_status": "scored",
        },
    ))
    store.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_batch_research(
        Namespace(asset="BTC", config=None, top=3, families="onchain,derivatives")
    )
    output = capsys.readouterr().out

    assert "Batch Research (BTC) [onchain,derivatives]" in output
    assert "Summary:  scored=1 retained=0 actionable=0 backed=1" in output
    assert "Top:      onchain_backed reason=backed" in output
    assert "price_dropped" not in output


def test_cmd_analyze_actionable_live_reports_signal_drop(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_actionable_live
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="backed",
        kind="dsl",
        definition={"expression": "btc_difficulty"},
        stake=0.2,
        source="random_dsl",
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_actionable_live": True,
            "lifecycle_signal_nonzero_ratio": 0.8,
            "lifecycle_signal_mean_abs": 0.2,
            "lifecycle_capital_eligible": True,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="signal_drop",
        kind="dsl",
        definition={"expression": "funding_rate_eth"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_actionable_live": False,
            "lifecycle_signal_nonzero_ratio": 0.05,
            "lifecycle_signal_mean_abs": 0.01,
        },
    ))
    store.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_actionable_live(Namespace(asset="BTC", config=None, top=3, families=None))
    output = capsys.readouterr().out

    assert "Actionable Live (BTC)" in output
    assert "Summary:  live_proven=2 actionable=1 backed=1" in output
    assert "Drop:     signal_both=1" in output
    assert "backed=1" in output
    assert "FamSig:   " in output
    assert "Signal:   min=0.20/0.05" in output
    assert "Top:      backed reason=backed" in output


def test_cmd_analyze_actionable_live_reports_redundancy_families(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_actionable_live
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="leader",
        kind="dsl",
        definition={"expression": "btc_difficulty"},
        stake=0.2,
        source="random_dsl",
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_actionable_live": True,
            "lifecycle_capital_eligible": True,
            "lifecycle_capital_backed": True,
            "lifecycle_signal_nonzero_ratio": 0.8,
            "lifecycle_signal_mean_abs": 0.2,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="shadow",
        kind="dsl",
        definition={"expression": "btc_difficulty"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_actionable_live": True,
            "lifecycle_capital_eligible": True,
            "lifecycle_capital_backed": False,
            "lifecycle_redundancy_capped_by": "leader",
            "lifecycle_signal_nonzero_ratio": 0.7,
            "lifecycle_signal_mean_abs": 0.1,
        },
    ))
    store.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_actionable_live(Namespace(asset="BTC", config=None, top=3, families=None))
    output = capsys.readouterr().out

    assert "RedFam:   onchain->onchain:1" in output


def test_cmd_analyze_actionable_live_filters_families(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_actionable_live
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="onchain_live",
        kind="dsl",
        definition={"expression": "btc_difficulty"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_actionable_live": False,
            "lifecycle_signal_nonzero_ratio": 0.1,
            "lifecycle_signal_mean_abs": 0.01,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="price_live",
        kind="dsl",
        definition={"expression": "fear_greed"},
        stake=0.0,
        source="random_dsl",
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_actionable_live": False,
            "lifecycle_signal_nonzero_ratio": 0.1,
            "lifecycle_signal_mean_abs": 0.01,
        },
    ))
    store.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_actionable_live(
        Namespace(asset="BTC", config=None, top=3, families="onchain,derivatives")
    )
    output = capsys.readouterr().out

    assert "Actionable Live (BTC) [onchain,derivatives]" in output
    assert "Summary:  live_proven=1 actionable=0 backed=0" in output
    assert "onchain_live" in output
    assert "price_live" not in output


def test_cmd_analyze_latest_combine_shows_cohort_breakdown(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_latest_combine
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore
    from alpha_os.paper.tracker import PaperPortfolioTracker, PortfolioSnapshot

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="boot",
        kind="technical",
        definition={"indicator": "x"},
        stake=1.0,
        metadata={
            "lifecycle_research_retained": True,
            "lifecycle_research_quality_source": "bootstrap_seed",
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="batch",
        kind="dsl",
        definition={"expression": "x"},
        stake=1.0,
        metadata={
            "lifecycle_research_retained": True,
            "lifecycle_research_quality_source": "batch_research_score",
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="serious",
        kind="dsl",
        definition={"expression": "z"},
        source="bootstrap_serious",
        stake=1.0,
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_actionable_live": True,
            "lifecycle_capital_backed": True,
            "lifecycle_research_quality_source": "bootstrap_seed",
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="live",
        kind="dsl",
        definition={"expression": "y"},
        stake=1.0,
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_research_quality_source": "batch_research_score",
        },
    ))
    store.close()

    tracker = PaperPortfolioTracker(db_path=tmp_path / "paper_trading.db")
    tracker.save_snapshot(PortfolioSnapshot(
        date="2026-03-23T00:00:00",
        cash=1000.0,
        positions={},
        portfolio_value=1000.0,
        daily_pnl=0.0,
        daily_return=0.0,
        combined_signal=0.1234,
        dd_scale=1.0,
        vol_scale=1.0,
        final_signal=0.1234,
    ))
    tracker.save_hypothesis_signals(
        "2026-03-23T00:00:00",
        {"boot": 0.3, "batch": -0.2, "serious": 0.25, "live": 0.1},
    )
    tracker.close()

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_latest_combine(Namespace(asset="BTC", config=None, top=4))
    output = capsys.readouterr().out

    assert "Latest Combine (BTC)" in output
    assert "Combined:  stored=+0.123400" in output
    assert "Snapshot:  selected=4 current_backed=4 dropped=0 missing=0" in output
    assert "Current:   nonzero=4 zero=0" in output
    assert "Cohorts:   bootstrap n=1/1" in output
    assert "serious n=1/1" in output
    assert "batch n=1/1" in output
    assert "live n=1/1" in output
    assert "Top:       boot cohort=bootstrap" in output
    assert "Top:       serious cohort=serious" in output
    assert "Top:       batch cohort=batch" in output
    assert "Top:       live cohort=live" in output


def test_cmd_analyze_latest_combine_counts_dropped_current_weights(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_latest_combine
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore
    from alpha_os.paper.tracker import PaperPortfolioTracker, PortfolioSnapshot

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="kept",
        kind="dsl",
        definition={"expression": "x"},
        stake=1.0,
        metadata={"lifecycle_live_proven": True},
    ))
    store.register(HypothesisRecord(
        hypothesis_id="dropped",
        kind="dsl",
        definition={"expression": "y"},
        stake=0.0,
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_capital_reason": "live_proven",
        },
    ))
    store.close()

    tracker = PaperPortfolioTracker(db_path=tmp_path / "paper_trading.db")
    tracker.save_snapshot(PortfolioSnapshot(
        date="2026-03-23T00:00:00",
        cash=1000.0,
        positions={},
        portfolio_value=1000.0,
        daily_pnl=0.0,
        daily_return=0.0,
        combined_signal=0.1234,
        dd_scale=1.0,
        vol_scale=1.0,
        final_signal=0.1234,
    ))
    tracker.save_hypothesis_signals(
        "2026-03-23T00:00:00",
        {"kept": 0.3, "dropped": -0.2},
    )
    tracker.close()

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_latest_combine(Namespace(asset="BTC", config=None, top=3))
    output = capsys.readouterr().out

    assert "Snapshot:  selected=2 current_backed=1 dropped=1 missing=0" in output
    assert "Current:   nonzero=1 zero=0" in output
    assert "Dropped:   live_proven=1" in output
    assert "Top:       kept cohort=live" in output
    assert "Top:       dropped" not in output


def test_cmd_analyze_latest_combine_reports_redundancy_capped_drop(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_latest_combine
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore
    from alpha_os.paper.tracker import PaperPortfolioTracker, PortfolioSnapshot

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="kept",
        kind="dsl",
        definition={"expression": "x"},
        stake=1.0,
        metadata={"lifecycle_live_proven": True},
    ))
    store.register(HypothesisRecord(
        hypothesis_id="capped",
        kind="dsl",
        definition={"expression": "y"},
        stake=0.0,
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_capital_reason": "live_proven",
            "lifecycle_redundancy_capped_by": "kept",
        },
    ))
    store.close()

    tracker = PaperPortfolioTracker(db_path=tmp_path / "paper_trading.db")
    tracker.save_snapshot(PortfolioSnapshot(
        date="2026-03-23T00:00:00",
        cash=1000.0,
        positions={},
        portfolio_value=1000.0,
        daily_pnl=0.0,
        daily_return=0.0,
        combined_signal=0.1234,
        dd_scale=1.0,
        vol_scale=1.0,
        final_signal=0.1234,
    ))
    tracker.save_hypothesis_signals(
        "2026-03-23T00:00:00",
        {"kept": 0.3, "capped": -0.2},
    )
    tracker.close()

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_latest_combine(Namespace(asset="BTC", config=None, top=3))
    output = capsys.readouterr().out

    assert "Snapshot:  selected=2 current_backed=1 dropped=1 missing=0" in output
    assert "Current:   nonzero=1 zero=0" in output
    assert "Dropped:   redundancy_capped=1" in output


def test_cmd_analyze_actionable_live_can_filter_by_source(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_actionable_live
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="serious_live",
        kind="dsl",
        definition={"expression": "x"},
        source="bootstrap_serious",
        stake=1.0,
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_actionable_live": True,
            "lifecycle_capital_backed": True,
            "lifecycle_signal_nonzero_ratio": 0.7,
            "lifecycle_signal_mean_abs": 0.4,
        },
    ))
    store.register(HypothesisRecord(
        hypothesis_id="batch_live",
        kind="dsl",
        definition={"expression": "y"},
        source="random_dsl",
        stake=0.0,
        metadata={
            "lifecycle_live_proven": True,
            "lifecycle_signal_nonzero_ratio": 0.0,
            "lifecycle_signal_mean_abs": 0.0,
        },
    ))
    store.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)

    cmd_analyze_actionable_live(
        Namespace(
            asset="BTC",
            config=None,
            top=5,
            families=None,
            source="bootstrap_serious",
        )
    )
    output = capsys.readouterr().out

    assert "Actionable Live (BTC) <bootstrap_serious>" in output
    assert "live_proven=1 actionable=1 backed=1" in output
    assert "batch_live" not in output


def test_cmd_rebalance_allocation_trust_dry_run_and_apply(monkeypatch, tmp_path, capsys):
    from argparse import Namespace

    from alpha_os.cli import cmd_rebalance_allocation_trust
    from alpha_os.forward.tracker import HypothesisObservationTracker
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="scored",
        kind="ml",
        definition={"model_ref": "m1"},
        stake=1.0,
        metadata={"oos_sharpe": 0.8},
    ))
    store.register(HypothesisRecord(
        hypothesis_id="unscored",
        kind="dsl",
        definition={"expression": "x"},
        stake=1.0,
    ))
    store.close()

    fwd = HypothesisObservationTracker(tmp_path / "hypothesis_observations.db")
    fwd.record("unscored", "2026-03-21", 1.0, 0.001)
    fwd.record("unscored", "2026-03-23", 1.0, 0.0012)
    fwd.close()

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)
    monkeypatch.setattr("alpha_os.cli.SIGNAL_CACHE_DB", tmp_path / "signal_cache.db")

    conn = sqlite3.connect(tmp_path / "signal_cache.db")
    conn.execute(
        "CREATE TABLE signals (name TEXT, date TEXT, value REAL, resolution TEXT DEFAULT '1d', PRIMARY KEY (name, date))"
    )
    conn.executemany(
        "INSERT INTO signals (name, date, value, resolution) VALUES (?, ?, ?, ?)",
        [
            ("btc_ohlcv", f"2026-01-{day:02d}", float(day), "1d")
            for day in range(1, 32)
        ],
    )
    conn.commit()
    conn.close()

    cmd_rebalance_allocation_trust(Namespace(asset="BTC", config=None, dry_run=True))
    output = capsys.readouterr().out
    assert "Allocation trust rebalance [DRY RUN]: asset=BTC" in output
    assert (
        "Summary: active=2 changed=2 zeroed=1 research_backed=1 live_proven=0 "
        "research_candidate_capped=0 redundancy_capped=0"
    ) in output

    store = HypothesisStore(hdb)
    assert store.get("scored").stake == pytest.approx(1.0)
    assert store.get("unscored").stake == pytest.approx(1.0)
    store.close()

    cmd_rebalance_allocation_trust(Namespace(asset="BTC", config=None, dry_run=False))
    output = capsys.readouterr().out
    assert "Allocation trust rebalance [APPLY]: asset=BTC" in output
    assert "Rebalance summary: updated=2 active=2 zeroed=1" in output

    store = HypothesisStore(hdb)
    assert store.get("unscored").stake == pytest.approx(0.0)
    assert store.get("scored").stake < 1.0
    store.close()


def test_cmd_rebalance_allocation_trust_caps_positive_correlation_bootstrap_pair(
    monkeypatch,
    tmp_path,
    capsys,
):
    from argparse import Namespace

    from alpha_os.cli import cmd_rebalance_allocation_trust
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    sdb = tmp_path / "signal_cache.db"
    store = HypothesisStore(hdb)
    for hypothesis_id in ("h_fast", "h_slow"):
        store.register(HypothesisRecord(
            hypothesis_id=hypothesis_id,
            kind="technical",
            definition={"indicator": "roc_momentum", "params": {"window": 20}},
            stake=1.0,
            source="bootstrap_technical",
            metadata={"oos_sharpe": 0.8 if hypothesis_id == "h_fast" else 0.7},
        ))
    store.close()

    conn = sqlite3.connect(sdb)
    conn.execute(
        "CREATE TABLE signals (name TEXT, date TEXT, value REAL, resolution TEXT DEFAULT '1d', PRIMARY KEY (name, date))"
    )
    conn.executemany(
        "INSERT INTO signals (name, date, value, resolution) VALUES (?, ?, ?, ?)",
        [
            ("btc_ohlcv", f"2026-01-{day:02d}", float(day), "1d")
            for day in range(1, 32)
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("alpha_os.cli.asset_data_dir", lambda asset: tmp_path)
    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)
    monkeypatch.setattr("alpha_os.cli.SIGNAL_CACHE_DB", sdb)

    cmd_rebalance_allocation_trust(Namespace(asset="BTC", config=None, dry_run=True))
    output = capsys.readouterr().out
    assert "redundancy_capped=1" in output
    assert "capped_by=h_fast" in output


def test_cmd_analyze_live_breadth_surfaces_redundant_bootstrap_pair(
    monkeypatch,
    tmp_path,
    capsys,
):
    from argparse import Namespace

    from alpha_os.cli import cmd_analyze_live_breadth
    from alpha_os.hypotheses.store import HypothesisRecord, HypothesisStore

    hdb = tmp_path / "hypotheses.db"
    sdb = tmp_path / "signal_cache.db"

    store = HypothesisStore(hdb)
    store.register(HypothesisRecord(
        hypothesis_id="technical_roc_20_momentum",
        kind="technical",
        definition={"indicator": "roc_momentum", "params": {"window": 20}},
        stake=0.5,
        source="bootstrap_technical",
        metadata={"lifecycle_bootstrap_trust": 0.1},
    ))
    store.register(HypothesisRecord(
        hypothesis_id="technical_volume_price_confirmation",
        kind="technical",
        definition={
            "indicator": "volume_price_confirmation",
            "params": {"price_window": 20, "volume_window": 20},
        },
        stake=0.4,
        source="bootstrap_technical",
        metadata={"lifecycle_bootstrap_trust": 0.1},
    ))
    store.close()

    conn = sqlite3.connect(sdb)
    conn.execute(
        "CREATE TABLE signals (name TEXT, date TEXT, value REAL, resolution TEXT DEFAULT '1d', PRIMARY KEY (name, date))"
    )
    conn.executemany(
        "INSERT INTO signals (name, date, value, resolution) VALUES (?, ?, ?, ?)",
        [
            ("btc_ohlcv", f"2026-01-{day:02d}", float(day), "1d")
            for day in range(1, 32)
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("alpha_os.cli.HYPOTHESES_DB", hdb)
    monkeypatch.setattr("alpha_os.cli.SIGNAL_CACHE_DB", sdb)

    cmd_analyze_live_breadth(
        Namespace(asset="BTC", config=None, lookback=30, top_pairs=3)
    )
    output = capsys.readouterr().out

    assert "Live breadth (BTC)" in output
    assert "records=2 analyzed=2 skipped=0 lookback=30" in output
    assert "technical_roc_20_momentum <-> technical_volume_price_confirmation" in output
    assert "corr=+1.000" in output
    assert "|corr|=1.000" in output
