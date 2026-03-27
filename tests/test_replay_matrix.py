from __future__ import annotations

from pathlib import Path

from alpha_os_recovery.experiments.matrix import load_replay_matrix, run_replay_matrix
from alpha_os_recovery.legacy.replay_experiment import ReplayExperimentRun


def test_load_replay_matrix_merges_defaults(tmp_path):
    manifest = tmp_path / "matrix.toml"
    manifest.write_text(
        """
[defaults]
asset = "BTC"
start_date = "2026-01-01"
end_date = "2026-01-31"
managed_alpha_mode = "admission"
deployment_mode = "refresh"

[defaults.overrides]
lifecycle.candidate_quality_min = 1.10

[[experiment]]
name = "baseline"

[[experiment]]
name = "deadband-10"

[experiment.overrides]
paper.rebalance_deadband_usd = 10.0
"""
    )

    matrix = load_replay_matrix(manifest)

    assert len(matrix.experiments) == 2
    assert matrix.experiments[0].name == "baseline"
    assert matrix.experiments[0].asset == "BTC"
    assert matrix.experiments[0].managed_alpha_mode == "admission"
    assert matrix.experiments[0].deployment_mode == "refresh"
    assert matrix.experiments[0].overrides == {
        "lifecycle.candidate_quality_min": 1.10,
    }
    assert matrix.experiments[1].overrides == {
        "lifecycle.candidate_quality_min": 1.10,
        "paper.rebalance_deadband_usd": 10.0,
    }


def test_run_replay_matrix_preserves_order(monkeypatch):
    calls: list[str] = []

    def fake_run(spec):
        calls.append(spec.name)
        return ReplayExperimentRun(
            experiment_id=spec.name,
            detail_path=Path(f"/tmp/{spec.name}.json"),
            index_path=Path("/tmp/index.jsonl"),
            payload={"name": spec.name, "result": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "total_trades": 0}},
        )

    monkeypatch.setattr("alpha_os_recovery.experiments.matrix.run_replay_experiment", fake_run)

    matrix = load_replay_matrix(
        Path(__file__).with_name("replay_matrix_fixture.toml")
    )
    runs = run_replay_matrix(matrix, max_workers=2)

    assert sorted(calls) == ["exp-a", "exp-b"]
    assert [run.experiment_id for run in runs] == ["exp-a", "exp-b"]
