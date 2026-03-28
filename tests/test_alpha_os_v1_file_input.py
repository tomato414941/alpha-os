from __future__ import annotations

import sqlite3
from pathlib import Path


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


def test_run_cycle_accepts_json_fixture(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    fixture_path = Path("tests/fixtures/v1_cycles/single_cycle.json")
    _register_hypothesis(main, db_path, "hyp_fixture")
    capsys.readouterr()

    rc = main(["apply-cycle", "--db", str(db_path), "--input", str(fixture_path)])
    assert rc == 0

    output = capsys.readouterr().out
    assert "Evaluation [created] BTC:residual_return_1d:2026-03-27" in output
    assert "Hyp:      hyp_fixture" in output

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT hypothesis_id, prediction_value, observation_value FROM evaluation_snapshots"
        ).fetchone()
        assert row == ("hyp_fixture", 0.4, 0.15)
    finally:
        conn.close()


def test_run_cycles_applies_multiple_fixture_days(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    fixture_path = Path("tests/fixtures/v1_cycles/multi_cycle.json")
    _register_hypothesis(main, db_path, "hyp_fixture")
    capsys.readouterr()

    rc = main(["apply-cycles", "--db", str(db_path), "--input", str(fixture_path)])
    assert rc == 0

    output = capsys.readouterr().out
    assert "Batch complete: evaluations=3 created=3 existing=0" in output
    assert "Latest:   BTC:residual_return_1d:2026-03-29 / hyp_fixture" in output

    conn = sqlite3.connect(db_path)
    try:
        snapshot_count = conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0]
        state_row = conn.execute(
            """
            SELECT prediction_count, observation_count
            FROM hypotheses
            WHERE hypothesis_id = 'hyp_fixture'
            """
        ).fetchone()
        assert snapshot_count == 3
        assert state_row == (3, 3)
    finally:
        conn.close()


def test_run_cycle_rejects_non_v1_asset_in_fixture(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    bad_fixture = tmp_path / "bad_cycle.json"
    bad_fixture.write_text(
        (
            "{\n"
            '  "date": "2026-03-30",\n'
            '  "hypothesis_id": "hyp_bad",\n'
            '  "prediction": 0.2,\n'
            '  "observation": 0.1,\n'
            '  "asset": "ETH"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    try:
        main(["apply-cycle", "--db", str(db_path), "--input", str(bad_fixture)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for non-v1 fixture asset")
