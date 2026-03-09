from __future__ import annotations

import json
from types import SimpleNamespace

from alpha_os.experiments.replay import (
    ReplayExperimentSpec,
    apply_config_overrides,
    parse_override_assignment,
    run_replay_experiment,
)
from alpha_os.alpha.registry import AlphaRecord, AlphaRegistry, AlphaState
from alpha_os.config import Config


def test_parse_override_assignment_uses_toml_types():
    key, value = parse_override_assignment("live_quality.weight_confidence_floor=0.25")

    assert key == "live_quality.weight_confidence_floor"
    assert value == 0.25


def test_apply_config_overrides_updates_nested_fields():
    cfg = Config()

    apply_config_overrides(
        cfg,
        {
            "lifecycle.candidate_quality_min": 1.1,
            "live_quality.weight_confidence_power": 4,
        },
    )

    assert cfg.lifecycle.candidate_quality_min == 1.1
    assert cfg.live_quality.weight_confidence_power == 4


def test_run_replay_experiment_writes_artifacts(tmp_path, monkeypatch):
    asset_root = tmp_path / "BTC"
    asset_root.mkdir(parents=True, exist_ok=True)
    registry = AlphaRegistry(asset_root / "alpha_registry.db")
    registry.register(
        AlphaRecord(
            alpha_id="a1",
            expression="x",
            state=AlphaState.ACTIVE,
            oos_sharpe=1.0,
        )
    )
    registry.close()

    monkeypatch.setattr(
        "alpha_os.experiments.replay.asset_data_dir",
        lambda asset: asset_root,
    )
    monkeypatch.setattr(
        "alpha_os.experiments.replay._git_commit",
        lambda: "deadbeef",
    )
    monkeypatch.setattr(
        "alpha_os.experiments.replay.run_replay",
        lambda **_: SimpleNamespace(
            n_days=10,
            initial_capital=10000.0,
            final_value=10123.0,
            total_return=0.0123,
            sharpe=1.5,
            max_drawdown=0.02,
            total_trades=7,
            n_skipped_deadband=1,
            n_skipped_min_notional=2,
            n_skipped_rounded_to_zero=3,
            win_rate=0.6,
            best_day=("2026-03-01", 0.01),
            worst_day=("2026-03-02", -0.01),
        ),
    )

    run = run_replay_experiment(
        ReplayExperimentSpec(
            name="smoke",
            asset="BTC",
            start_date="2026-03-01",
            end_date="2026-03-05",
            deployment_mode="refresh",
            overrides={"lifecycle.candidate_quality_min": 1.1},
        )
    )

    assert run.detail_path.exists()
    assert run.index_path.exists()

    payload = json.loads(run.detail_path.read_text())
    summary = json.loads(run.index_path.read_text().strip().splitlines()[-1])

    assert payload["name"] == "smoke"
    assert payload["git_commit"] == "deadbeef"
    assert payload["spec"]["deployment_mode"] == "refresh"
    assert payload["overrides"]["lifecycle.candidate_quality_min"] == 1.1
    assert payload["result"]["final_value"] == 10123.0
    assert payload["result"]["n_skipped_deadband"] == 1
    assert payload["result"]["n_skipped_min_notional"] == 2
    assert payload["result"]["n_skipped_rounded_to_zero"] == 3
    assert summary["detail_path"] == str(run.detail_path)
