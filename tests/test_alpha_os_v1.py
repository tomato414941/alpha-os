from __future__ import annotations

import sqlite3


def _register_hypothesis(main, db_path, hypothesis_id: str) -> None:
    assert (
        main(
            [
                "register-hypothesis",
                "--db",
                str(db_path),
                "--hypothesis-id",
                hypothesis_id,
            ]
        )
        == 0
    )


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
            "run-cycle",
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
    assert "Cycle [created] BTC:residual_return_1d:2026-03-26" in output
    assert "alpha-os v1 status" in output
    assert "Live:     1" in output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT quality_after, quality_delta,
                   allocation_trust_after, allocation_trust_delta, generated_weight,
                   input_source
            FROM cycle_snapshots
            """
        ).fetchone()
        assert row is not None
        assert row[0] > 0.0
        assert row[1] > 0.0
        assert row[2] > 0.0
        assert row[3] > 0.0
        assert row[4] == 1.0
        assert row[5] == "manual"
    finally:
        conn.close()


def test_run_cycle_is_idempotent_for_same_cycle_id(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    capsys.readouterr()
    args = [
        "run-cycle",
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
    assert "Cycle [created]" in first_output

    assert main(args) == 0
    second_output = capsys.readouterr().out
    assert "Cycle [existing]" in second_output

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM cycle_snapshots").fetchone()[0]
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

    _register_hypothesis(main, db_path, "hyp_1")
    first_output = capsys.readouterr().out
    assert "Hypothesis [created] hyp_1" in first_output

    _register_hypothesis(main, db_path, "hyp_1")
    second_output = capsys.readouterr().out
    assert "Hypothesis [existing] hyp_1" in second_output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT status, quality, allocation_trust, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'hyp_1'
            """
        ).fetchone()
        assert row == ("registered", 0.0, 0.0, 0, 0)
    finally:
        conn.close()


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
    assert "Prediction [created] BTC:residual_return_1d:2026-03-26" in first_output

    assert main(args) == 0
    second_output = capsys.readouterr().out
    assert "Prediction [existing] BTC:residual_return_1d:2026-03-26" in second_output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT value
            FROM predictions
            WHERE cycle_id = 'BTC:residual_return_1d:2026-03-26'
              AND hypothesis_id = 'hyp_1'
            """
        ).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert row == (0.5,)
        assert count == 1
    finally:
        conn.close()


def test_record_prediction_rejects_unregistered_hypothesis(tmp_path):
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
        raise AssertionError("expected parser exit for unregistered hypothesis")


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
    assert "Observation [created] BTC:residual_return_1d:2026-03-26" in first_output

    assert main(args) == 0
    second_output = capsys.readouterr().out
    assert "Observation [existing] BTC:residual_return_1d:2026-03-26" in second_output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT value
            FROM observations
            WHERE cycle_id = 'BTC:residual_return_1d:2026-03-26'
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
                "run-cycle",
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
    assert "Cycle [created] BTC:residual_return_1d:2026-03-26" in output

    conn = sqlite3.connect(db_path)
    try:
        prediction_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        snapshot_count = conn.execute("SELECT COUNT(*) FROM cycle_snapshots").fetchone()[0]
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
                "run-cycle",
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
    assert "Cycle [created] BTC:residual_return_1d:2026-03-26" in output

    conn = sqlite3.connect(db_path)
    try:
        observation_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        snapshot_count = conn.execute("SELECT COUNT(*) FROM cycle_snapshots").fetchone()[0]
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
    assert "Cycle [created] BTC:residual_return_1d:2026-03-26" in output

    conn = sqlite3.connect(db_path)
    try:
        snapshot_count = conn.execute("SELECT COUNT(*) FROM cycle_snapshots").fetchone()[0]
        state_row = conn.execute(
            """
            SELECT status, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'hyp_1'
            """
        ).fetchone()
        assert snapshot_count == 1
        assert state_row == ("live", 1, 1)
    finally:
        conn.close()


def test_update_state_is_idempotent_for_same_cycle(tmp_path, capsys):
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
    assert "Cycle [created]" in first_output

    assert main(args) == 0
    second_output = capsys.readouterr().out
    assert "Cycle [existing]" in second_output

    conn = sqlite3.connect(db_path)
    try:
        snapshot_count = conn.execute("SELECT COUNT(*) FROM cycle_snapshots").fetchone()[0]
        state_row = conn.execute(
            """
            SELECT status, prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'hyp_1'
            """
        ).fetchone()
        assert snapshot_count == 1
        assert state_row == ("live", 1, 1)
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
    assert "Prediction [created] BTC:residual_return_1d:2026-03-26" in prediction_output

    _finalize_observation(main, db_path, "2026-03-26", "0.2")
    observation_output = capsys.readouterr().out
    assert "Observation [created] BTC:residual_return_1d:2026-03-26" in observation_output

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
    assert "Cycle [created] BTC:residual_return_1d:2026-03-26" in update_output

    assert main(["status", "--db", str(db_path)]) == 0
    status_output = capsys.readouterr().out
    assert "alpha-os v1 status" in status_output
    assert "Latest:   BTC:residual_return_1d:2026-03-26" in status_output
    assert "Trust:    total=" in status_output

    assert main(["show-cycles", "--db", str(db_path), "--limit", "5"]) == 0
    cycles_output = capsys.readouterr().out
    assert "alpha-os v1 cycles" in cycles_output
    assert "Count:    1" in cycles_output
    assert "source=-" in cycles_output

    conn = sqlite3.connect(db_path)
    try:
        counts = {
            "hypotheses": conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0],
            "predictions": conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            "observations": conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
            "snapshots": conn.execute("SELECT COUNT(*) FROM cycle_snapshots").fetchone()[0],
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
        assert status_row == ("live",)
    finally:
        conn.close()


def test_pause_resume_and_retire_hypothesis_follow_state_machine(tmp_path, capsys):
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

    assert main(["pause-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"]) == 0
    pause_output = capsys.readouterr().out
    assert "Hypothesis [paused] hyp_1" in pause_output

    assert main(["resume-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"]) == 0
    resume_output = capsys.readouterr().out
    assert "Hypothesis [resumed] hyp_1" in resume_output

    assert main(["retire-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"]) == 0
    retire_output = capsys.readouterr().out
    assert "Hypothesis [retired] hyp_1" in retire_output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT status FROM hypotheses WHERE hypothesis_id = 'hyp_1'"
        ).fetchone()
        assert row == ("retired",)
    finally:
        conn.close()


def test_pause_rejects_non_live_hypothesis(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")

    try:
        main(["pause-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for invalid pause transition")


def test_resume_rejects_non_paused_hypothesis(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")

    try:
        main(["resume-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for invalid resume transition")


def test_retired_hypothesis_rejects_new_prediction_and_update(tmp_path):
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
    assert main(["pause-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"]) == 0
    assert main(["retire-hypothesis", "--db", str(db_path), "--hypothesis-id", "hyp_1"]) == 0

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
        raise AssertionError("expected parser exit for retired prediction record")

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
        raise AssertionError("expected parser exit for retired update")


def test_run_cycle_rejects_mismatched_pre_recorded_observation(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    _register_hypothesis(main, db_path, "hyp_1")
    _finalize_observation(main, db_path, "2026-03-26", "0.2")

    try:
        main(
            [
                "run-cycle",
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


def test_run_cycle_rejects_unregistered_hypothesis(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    try:
        main(
            [
                "run-cycle",
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
        raise AssertionError("expected parser exit for unregistered hypothesis")


def test_init_db_creates_empty_runtime(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    assert main(["init-db", "--db", str(db_path)]) == 0
    assert main(["status", "--db", str(db_path)]) == 0

    output = capsys.readouterr().out
    assert "Initialized v1 db" in output
    assert "no cycles recorded" in output
