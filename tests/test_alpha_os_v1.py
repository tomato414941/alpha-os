from __future__ import annotations

import json
import sqlite3


def _register_hypothesis(main, db_path, hypothesis_id: str, *, target_id: str | None = None) -> None:
    argv = [
        "register-hypothesis",
        "--db",
        str(db_path),
        "--hypothesis-id",
        hypothesis_id,
    ]
    if target_id is not None:
        argv.extend(["--target-id", target_id])
    assert main(argv) == 0


def _finalize_observation(main, db_path, date: str, observation: str) -> None:
    assert (
        main(
            [
                "finalize-observation",
                "--db",
                str(db_path),
                "--date",
                date,
                "--observation",
                observation,
            ]
        )
        == 0
    )


def _record_prediction(main, db_path, date: str, hypothesis_id: str, prediction: str) -> None:
    assert (
        main(
            [
                "record-prediction",
                "--db",
                str(db_path),
                "--date",
                date,
                "--hypothesis-id",
                hypothesis_id,
                "--prediction",
                prediction,
            ]
        )
        == 0
    )


def test_run_cycle_writes_snapshot_and_status(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    capsys.readouterr()
    rc = main(
        [
            "apply-evaluation",
            "--db",
            str(db_path),
            "--date",
            "2026-03-26",
            "--hypothesis-id",
            "hyp_1",
            "--prediction",
            "0.5",
            "--observation",
            "0.2",
        ]
    )
    assert rc == 0

    status_rc = main(["status", "--db", str(db_path)])
    assert status_rc == 0
    output = capsys.readouterr().out
    assert "Evaluation [created] BTC:residual_return_3d:2026-03-26" in output
    assert "alpha-os status" in output
    assert "Metrics:  tracked=1" in output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT prediction_value, observation_value, signed_edge, absolute_error, input_source
            FROM evaluation_snapshots
            """
        ).fetchone()
        assert row is not None
        assert row[0] == 0.5
        assert row[1] == 0.2
        assert row[2] == 0.1
        assert row[3] == 0.3
        assert row[4] == "manual"
    finally:
        conn.close()


def test_run_cycle_is_idempotent_for_same_evaluation_id(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    capsys.readouterr()
    args = [
        "apply-evaluation",
        "--db",
        str(db_path),
        "--date",
        "2026-03-26",
        "--hypothesis-id",
        "hyp_1",
        "--prediction",
        "0.25",
        "--observation",
        "0.1",
    ]

    assert main(args) == 0
    first_output = capsys.readouterr().out
    assert "Evaluation [created]" in first_output

    assert main(args) == 0
    second_output = capsys.readouterr().out
    assert "Evaluation [existing]" in second_output

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0]
        prediction_count = conn.execute(
            "SELECT prediction_count FROM hypotheses WHERE hypothesis_id = 'hyp_1'"
        ).fetchone()[0]
        assert count == 1
        assert prediction_count == 1
    finally:
        conn.close()


def test_register_hypothesis_creates_state_and_is_idempotent(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    _register_hypothesis(main, db_path, "momentum_1d")
    first_output = capsys.readouterr().out
    assert "Hypothesis [created] momentum_1d" in first_output
    assert "Kind:     momentum" in first_output
    assert "Signal:   btc_ohlcv" in first_output
    assert "Lookback: 1" in first_output

    _register_hypothesis(main, db_path, "momentum_1d")
    second_output = capsys.readouterr().out
    assert "Hypothesis [existing] momentum_1d" in second_output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT definition_json, status, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'momentum_1d'
            """
        ).fetchone()
        assert row is not None
        definition = json.loads(row[0])
        assert definition == {
            "kind": "momentum",
            "signal_name": "btc_ohlcv",
            "target_definition": {
                "family": "residual_return",
                "observation_kind": "fixed_horizon",
                "output_kind": "real_value",
                "params": {"horizon_days": 3},
                "scoring_kind": "corr_mmc",
                "subject_kind": "asset",
                "target_id": "residual_return_3d",
            },
            "params": {"lookback": 1},
        }
        assert row[1:] == ("active", 0, 0)
    finally:
        conn.close()


def test_register_hypothesis_registers_target_definition(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    _register_hypothesis(main, db_path, "momentum_1d")
    capsys.readouterr()

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT target_id, definition_json
            FROM targets
            WHERE target_id = 'residual_return_3d'
            """
        ).fetchone()
        assert row is not None
        definition = json.loads(row[1])
        assert row[0] == "residual_return_3d"
        assert definition == {
            "target_id": "residual_return_3d",
            "family": "residual_return",
            "observation_kind": "fixed_horizon",
            "subject_kind": "asset",
            "output_kind": "real_value",
            "scoring_kind": "corr_mmc",
            "params": {"horizon_days": 3},
        }
    finally:
        conn.close()


def test_register_hypothesis_supports_new_builtin_definition(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    _register_hypothesis(main, db_path, "reversal_5d")
    output = capsys.readouterr().out
    assert "Hypothesis [created] reversal_5d" in output
    assert "Kind:     reversal" in output
    assert "Signal:   btc_ohlcv" in output
    assert "Lookback: 5" in output
    assert "Horizon:  3d" in output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT definition_json, status, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'reversal_5d'
            """
        ).fetchone()
        assert row is not None
        definition = json.loads(row[0])
        assert definition == {
            "kind": "reversal",
            "signal_name": "btc_ohlcv",
            "target_definition": {
                "family": "residual_return",
                "observation_kind": "fixed_horizon",
                "output_kind": "real_value",
                "params": {"horizon_days": 3},
                "scoring_kind": "corr_mmc",
                "subject_kind": "asset",
                "target_id": "residual_return_3d",
            },
            "params": {"lookback": 5},
        }
        assert row[1:] == ("active", 0, 0)
    finally:
        conn.close()


def test_register_hypothesis_supports_additional_kind_definition(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    _register_hypothesis(main, db_path, "average_gap_3d")
    output = capsys.readouterr().out
    assert "Hypothesis [created] average_gap_3d" in output
    assert "Kind:     average_gap" in output
    assert "Signal:   btc_ohlcv" in output
    assert "Lookback: 3" in output
    assert "Horizon:  3d" in output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT definition_json, status, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'average_gap_3d'
            """
        ).fetchone()
        assert row is not None
        definition = json.loads(row[0])
        assert definition == {
            "kind": "average_gap",
            "signal_name": "btc_ohlcv",
            "target_definition": {
                "family": "residual_return",
                "observation_kind": "fixed_horizon",
                "output_kind": "real_value",
                "params": {"horizon_days": 3},
                "scoring_kind": "corr_mmc",
                "subject_kind": "asset",
                "target_id": "residual_return_3d",
            },
            "params": {"lookback": 3},
        }
        assert row[1:] == ("active", 0, 0)
    finally:
        conn.close()


def test_register_hypothesis_keeps_unknown_definition_nullable(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    _register_hypothesis(main, db_path, "hyp_1")
    output = capsys.readouterr().out
    assert "Hypothesis [created] hyp_1" in output
    assert "Kind:" not in output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT definition_json, status
            FROM hypotheses
            WHERE hypothesis_id = 'hyp_1'
            """
        ).fetchone()
        assert row == (None, "active")
    finally:
        conn.close()


def test_register_hypothesis_rejects_mismatched_builtin_target(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    try:
        main(
            [
                "register-hypothesis",
                "--db",
                str(db_path),
                "--hypothesis-id",
                "momentum_1d",
                "--target-id",
                "residual_return_1d",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for mismatched built-in target")


def test_record_prediction_creates_row_and_is_idempotent(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    capsys.readouterr()

    args = [
        "record-prediction",
        "--db",
        str(db_path),
        "--date",
        "2026-03-26",
        "--hypothesis-id",
        "hyp_1",
        "--prediction",
        "0.5",
    ]

    assert main(args) == 0
    first_output = capsys.readouterr().out
    assert "Prediction [created] BTC:residual_return_3d:2026-03-26" in first_output

    assert main(args) == 0
    second_output = capsys.readouterr().out
    assert "Prediction [existing] BTC:residual_return_3d:2026-03-26" in second_output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT value
            FROM predictions
            WHERE evaluation_id = 'BTC:residual_return_3d:2026-03-26'
              AND hypothesis_id = 'hyp_1'
            """
        ).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert row == (0.5,)
        assert count == 1
    finally:
        conn.close()


def test_record_prediction_rejects_unknown_hypothesis(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    try:
        main(
            [
                "record-prediction",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
                "--prediction",
                "0.5",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for unknown hypothesis")


def test_finalize_observation_creates_row_and_is_idempotent(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    args = [
        "finalize-observation",
        "--db",
        str(db_path),
        "--date",
        "2026-03-26",
        "--observation",
        "0.2",
    ]

    assert main(args) == 0
    first_output = capsys.readouterr().out
    assert "Observation [created] BTC:residual_return_3d:2026-03-26" in first_output

    assert main(args) == 0
    second_output = capsys.readouterr().out
    assert "Observation [existing] BTC:residual_return_3d:2026-03-26" in second_output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT value
            FROM observations
            WHERE evaluation_id = 'BTC:residual_return_3d:2026-03-26'
            """
        ).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        assert row == (0.2,)
        assert count == 1
    finally:
        conn.close()


def test_run_cycle_reuses_pre_recorded_prediction(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    capsys.readouterr()
    _record_prediction(main, db_path, "2026-03-26", "hyp_1", "0.5")
    capsys.readouterr()

    assert (
        main(
            [
                "apply-evaluation",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
                "--prediction",
                "0.5",
                "--observation",
                "0.2",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Evaluation [created] BTC:residual_return_3d:2026-03-26" in output

    conn = sqlite3.connect(db_path)
    try:
        prediction_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        snapshot_count = conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0]
        assert prediction_count == 1
        assert snapshot_count == 1
    finally:
        conn.close()


def test_run_cycle_reuses_pre_recorded_observation(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    capsys.readouterr()
    _finalize_observation(main, db_path, "2026-03-26", "0.2")
    capsys.readouterr()

    assert (
        main(
            [
                "apply-evaluation",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
                "--prediction",
                "0.5",
                "--observation",
                "0.2",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Evaluation [created] BTC:residual_return_3d:2026-03-26" in output

    conn = sqlite3.connect(db_path)
    try:
        observation_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        snapshot_count = conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0]
        assert observation_count == 1
        assert snapshot_count == 1
    finally:
        conn.close()


def test_update_state_uses_recorded_prediction_and_observation(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    _record_prediction(main, db_path, "2026-03-26", "hyp_1", "0.5")
    _finalize_observation(main, db_path, "2026-03-26", "0.2")
    capsys.readouterr()

    assert (
        main(
            [
                "update-state",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Evaluation [created] BTC:residual_return_3d:2026-03-26" in output

    conn = sqlite3.connect(db_path)
    try:
        snapshot_count = conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0]
        state_row = conn.execute(
            """
            SELECT status, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'hyp_1'
            """
        ).fetchone()
        assert snapshot_count == 1
        assert state_row == ("active", 1, 1)
    finally:
        conn.close()


def test_update_state_is_idempotent_for_same_evaluation(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    _record_prediction(main, db_path, "2026-03-26", "hyp_1", "0.5")
    _finalize_observation(main, db_path, "2026-03-26", "0.2")
    capsys.readouterr()

    args = [
        "update-state",
        "--db",
        str(db_path),
        "--date",
        "2026-03-26",
        "--hypothesis-id",
        "hyp_1",
    ]
    assert main(args) == 0
    first_output = capsys.readouterr().out
    assert "Evaluation [created]" in first_output

    assert main(args) == 0
    second_output = capsys.readouterr().out
    assert "Evaluation [existing]" in second_output

    conn = sqlite3.connect(db_path)
    try:
        snapshot_count = conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0]
        state_row = conn.execute(
            """
            SELECT status, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'hyp_1'
            """
        ).fetchone()
        assert snapshot_count == 1
        assert state_row == ("active", 1, 1)
    finally:
        conn.close()


def test_same_evaluation_can_record_multiple_hypothesis_results(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "momentum_1d")
    _register_hypothesis(main, db_path, "reversal_1d")
    capsys.readouterr()

    assert (
        main(
            [
                "apply-evaluation",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "momentum_1d",
                "--prediction",
                "0.25",
                "--observation",
                "0.1",
            ]
        )
        == 0
    )
    first_output = capsys.readouterr().out
    assert "Evaluation [created] BTC:residual_return_3d:2026-03-26" in first_output
    assert "Hyp:      momentum_1d" in first_output

    assert (
        main(
            [
                "apply-evaluation",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "reversal_1d",
                "--prediction",
                "-0.25",
                "--observation",
                "0.1",
            ]
        )
        == 0
    )
    second_output = capsys.readouterr().out
    assert "Evaluation [created] BTC:residual_return_3d:2026-03-26" in second_output
    assert "Hyp:      reversal_1d" in second_output

    conn = sqlite3.connect(db_path)
    try:
        prediction_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        observation_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        snapshot_count = conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0]
        rows = conn.execute(
            """
            SELECT evaluation_id, hypothesis_id
            FROM evaluation_snapshots
            ORDER BY hypothesis_id
            """
        ).fetchall()
        assert prediction_count == 2
        assert observation_count == 1
        assert snapshot_count == 2
        assert rows == [
            ("BTC:residual_return_3d:2026-03-26", "momentum_1d"),
            ("BTC:residual_return_3d:2026-03-26", "reversal_1d"),
        ]
    finally:
        conn.close()


def test_update_state_rejects_missing_prediction(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    _finalize_observation(main, db_path, "2026-03-26", "0.2")

    try:
        main(
            [
                "update-state",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for missing prediction")


def test_update_state_rejects_missing_observation(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    _record_prediction(main, db_path, "2026-03-26", "hyp_1", "0.5")

    try:
        main(
            [
                "update-state",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for missing observation")


def test_v2_smoke_flow_registers_records_finalizes_and_updates(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    _register_hypothesis(main, db_path, "hyp_1")
    register_output = capsys.readouterr().out
    assert "Hypothesis [created] hyp_1" in register_output

    _record_prediction(main, db_path, "2026-03-26", "hyp_1", "0.5")
    prediction_output = capsys.readouterr().out
    assert "Prediction [created] BTC:residual_return_3d:2026-03-26" in prediction_output

    _finalize_observation(main, db_path, "2026-03-26", "0.2")
    observation_output = capsys.readouterr().out
    assert "Observation [created] BTC:residual_return_3d:2026-03-26" in observation_output

    assert (
        main(
            [
                "update-state",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
            ]
        )
        == 0
    )
    update_output = capsys.readouterr().out
    assert "Evaluation [created] BTC:residual_return_3d:2026-03-26" in update_output

    assert main(["status", "--db", str(db_path)]) == 0
    status_output = capsys.readouterr().out
    assert "alpha-os status" in status_output
    assert "Latest:   BTC:residual_return_3d:2026-03-26 / hyp_1" in status_output
    assert "Metrics:  tracked=1" in status_output

    assert main(["show-evaluations", "--db", str(db_path), "--limit", "5"]) == 0
    cycles_output = capsys.readouterr().out
    assert "alpha-os evaluations" in cycles_output
    assert "Count:    1" in cycles_output
    assert "source=-" in cycles_output

    conn = sqlite3.connect(db_path)
    try:
        counts = {
            "hypotheses": conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0],
            "predictions": conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            "observations": conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
            "snapshots": conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0],
        }
        status_row = conn.execute(
            "SELECT status FROM hypotheses WHERE hypothesis_id = 'hyp_1'"
        ).fetchone()
        assert counts == {
            "hypotheses": 1,
            "predictions": 1,
            "observations": 1,
            "snapshots": 1,
        }
        assert status_row == ("active",)
    finally:
        conn.close()


def test_activate_and_deactivate_hypothesis_follow_state_machine(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    _record_prediction(main, db_path, "2026-03-26", "hyp_1", "0.5")
    _finalize_observation(main, db_path, "2026-03-26", "0.2")
    assert (
        main(
            [
                "update-state",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(["deactivate-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"])
        == 0
    )
    deactivate_output = capsys.readouterr().out
    assert "Hypothesis [deactivated] hyp_1" in deactivate_output

    assert (
        main(["activate-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"])
        == 0
    )
    activate_output = capsys.readouterr().out
    assert "Hypothesis [activated] hyp_1" in activate_output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT status FROM hypotheses WHERE hypothesis_id = 'hyp_1'"
        ).fetchone()
        assert row == ("active",)
    finally:
        conn.close()


def test_deactivate_allows_active_hypothesis(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    capsys.readouterr()

    assert (
        main(["deactivate-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"])
        == 0
    )
    output = capsys.readouterr().out
    assert "Hypothesis [deactivated] hyp_1" in output


def test_activate_rejects_non_inactive_hypothesis(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")

    try:
        main(["activate-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for invalid activate transition")


def test_inactive_hypothesis_rejects_new_prediction_and_update(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    _record_prediction(main, db_path, "2026-03-26", "hyp_1", "0.5")
    _finalize_observation(main, db_path, "2026-03-26", "0.2")
    assert (
        main(
            [
                "update-state",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
            ]
        )
        == 0
    )
    assert (
        main(["deactivate-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"])
        == 0
    )

    try:
        main(
            [
                "record-prediction",
                "--db",
                str(db_path),
                "--date",
                "2026-03-27",
                "--hypothesis-id",
                "hyp_1",
                "--prediction",
                "0.1",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for inactive prediction record")

    try:
        main(
            [
                "update-state",
                "--db",
                str(db_path),
                "--date",
                "2026-03-27",
                "--hypothesis-id",
                "hyp_1",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for inactive update")


def test_run_cycle_rejects_mismatched_pre_recorded_observation(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    _finalize_observation(main, db_path, "2026-03-26", "0.2")

    try:
        main(
            [
                "apply-evaluation",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
                "--prediction",
                "0.5",
                "--observation",
                "0.1",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for mismatched pre-recorded observation")


def test_apply_evaluation_rejects_unknown_hypothesis(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    try:
        main(
            [
                "apply-evaluation",
                "--db",
                str(db_path),
                "--date",
                "2026-03-26",
                "--hypothesis-id",
                "hyp_1",
                "--prediction",
                "0.5",
                "--observation",
                "0.2",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for unknown hypothesis")


def test_init_db_creates_empty_runtime(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    assert main(["init-db", "--db", str(db_path)]) == 0
    assert main(["status", "--db", str(db_path)]) == 0

    output = capsys.readouterr().out
    assert "Initialized runtime db" in output
    assert "no evaluations recorded" in output
