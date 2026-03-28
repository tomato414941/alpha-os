from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


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


def test_generate_evaluation_input_from_frame_uses_prev_and_next_close():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 121.0},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        hypothesis_id="momentum_1d",
    )

    assert evaluation_input.date == "2026-03-27"
    assert evaluation_input.hypothesis_id == "momentum_1d"
    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx(0.1)


def test_generate_evaluation_input_rejects_missing_neighbor_rows():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 121.0},
        ]
    )

    try:
        generate_evaluation_input_from_frame(
            frame=frame,
            date="2026-03-27",
            hypothesis_id="momentum_1d",
        )
    except ValueError as exc:
        assert "prior daily returns" in str(exc)
    else:
        raise AssertionError("expected previous-close validation error")


def test_generate_evaluation_inputs_from_frame_uses_date_range():
    from alpha_os.evaluation_generation import generate_evaluation_inputs_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 105.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 115.5},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 121.275},
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 127.33875},
        ]
    )

    evaluation_inputs = generate_evaluation_inputs_from_frame(
        frame=frame,
        start_date="2026-03-26",
        end_date="2026-03-28",
        hypothesis_id="momentum_1d",
    )

    assert [item.date for item in evaluation_inputs] == [
        "2026-03-26",
        "2026-03-27",
        "2026-03-28",
    ]
    assert all(item.hypothesis_id == "momentum_1d" for item in evaluation_inputs)


def test_generate_evaluation_input_from_frame_supports_momentum_3d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 146.41},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        hypothesis_id="momentum_3d",
    )

    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx(0.1)


def test_generate_evaluation_input_from_frame_supports_momentum_5d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-22T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-23T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 146.41},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 161.051},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 177.1561},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        hypothesis_id="momentum_5d",
    )

    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx(0.1)


def test_generate_evaluation_input_from_frame_supports_reversal_1d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 121.0},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        hypothesis_id="reversal_1d",
    )

    assert evaluation_input.prediction == pytest.approx(-0.1)
    assert evaluation_input.observation == pytest.approx(0.1)


def test_generate_evaluation_input_from_frame_supports_reversal_3d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 146.41},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        hypothesis_id="reversal_3d",
    )

    assert evaluation_input.prediction == pytest.approx(-0.1)
    assert evaluation_input.observation == pytest.approx(0.1)


def test_generate_evaluation_input_from_frame_supports_average_gap_3d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 146.41},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        hypothesis_id="average_gap_3d",
    )

    expected = (133.1 / ((110.0 + 121.0 + 133.1) / 3.0)) - 1.0
    assert evaluation_input.prediction == pytest.approx(expected)
    assert evaluation_input.observation == pytest.approx(0.1)


def test_generate_evaluation_input_from_frame_supports_range_position_5d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-22T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-23T00:00:00+00:00", "close": 104.0},
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 102.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 108.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 106.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 111.0},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        hypothesis_id="range_position_5d",
    )

    window = [104.0, 102.0, 108.0, 106.0, 110.0]
    expected = ((110.0 - min(window)) / (max(window) - min(window))) * 2.0 - 1.0
    assert evaluation_input.prediction == pytest.approx(expected)
    assert evaluation_input.observation == pytest.approx((111.0 / 110.0) - 1.0)


def test_generate_evaluation_input_from_signal_noise_uses_value_series(monkeypatch):
    from alpha_os.evaluation_generation import generate_evaluation_input_from_signal_noise

    calls: list[tuple[str, str]] = []

    class FakeClient:
        def get_data(self, name: str, since: str | None = None, resolution: str | None = None):
            calls.append((name, resolution or ""))
            return pd.DataFrame(
                [
                    {"timestamp": "2026-03-26T00:00:00+00:00", "value": 100.0},
                    {"timestamp": "2026-03-27T00:00:00+00:00", "value": 110.0},
                    {"timestamp": "2026-03-28T00:00:00+00:00", "value": 121.0},
                ]
            )

    monkeypatch.setattr(
        "alpha_os.evaluation_generation.build_signal_client",
        lambda **_kwargs: FakeClient(),
    )

    evaluation_input = generate_evaluation_input_from_signal_noise(
        date="2026-03-27",
        hypothesis_id="momentum_1d",
        base_url="https://signal-noise.example",
    )

    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx(0.1)
    assert calls == [("btc_ohlcv", "1d")]


def test_cmd_generate_evaluation_input_writes_json(tmp_path, monkeypatch, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"
    output_path = tmp_path / "cycle.json"

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_input_from_signal_noise",
        lambda **_kwargs: EvaluationInput(
            date="2026-03-27",
            hypothesis_id="momentum_1d",
            prediction=0.05,
            observation=-0.02,
        ),
    )
    _register_hypothesis(main, db_path, "momentum_1d")
    capsys.readouterr()

    rc = main(
        [
            "generate-evaluation-input",
            "--db",
            str(db_path),
            "--date",
            "2026-03-27",
            "--hypothesis-id",
            "momentum_1d",
            "--out",
            str(output_path),
        ]
    )
    assert rc == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-03-27"
    assert payload["hypothesis_id"] == "momentum_1d"
    assert payload["prediction"] == 0.05
    assert payload["observation"] == -0.02

    output = capsys.readouterr().out
    assert "Generated evaluation input:" in output
    assert "Signal:   pred=0.050000 obs=-0.020000" in output


def test_cmd_generate_evaluation_inputs_writes_json_array(tmp_path, monkeypatch, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"
    output_path = tmp_path / "cycles.json"

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_inputs_from_signal_noise",
        lambda **_kwargs: [
            EvaluationInput(
                date="2026-03-27",
                hypothesis_id="momentum_1d",
                prediction=0.05,
                observation=-0.02,
            ),
            EvaluationInput(
                date="2026-03-28",
                hypothesis_id="momentum_1d",
                prediction=-0.02,
                observation=0.03,
            ),
        ],
    )
    _register_hypothesis(main, db_path, "momentum_1d")
    capsys.readouterr()

    rc = main(
        [
            "generate-evaluation-inputs",
            "--db",
            str(db_path),
            "--start-date",
            "2026-03-27",
            "--end-date",
            "2026-03-28",
            "--hypothesis-id",
            "momentum_1d",
            "--out",
            str(output_path),
        ]
    )
    assert rc == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0]["date"] == "2026-03-27"
    assert payload[1]["date"] == "2026-03-28"

    output = capsys.readouterr().out
    assert "Generated evaluation inputs:" in output
    assert "Count:    2" in output


def test_generated_evaluation_input_can_feed_apply_cycle(tmp_path, monkeypatch):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"
    input_path = tmp_path / "cycle.json"

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_input_from_signal_noise",
        lambda **_kwargs: EvaluationInput(
            date="2026-03-27",
            hypothesis_id="momentum_1d",
            prediction=0.05,
            observation=-0.02,
        ),
    )
    _register_hypothesis(main, db_path, "momentum_1d")

    assert (
        main(
            [
                "generate-evaluation-input",
                "--db",
                str(db_path),
                "--date",
                "2026-03-27",
                "--hypothesis-id",
                "momentum_1d",
                "--out",
                str(input_path),
            ]
        )
        == 0
    )
    assert main(["apply-evaluation", "--db", str(db_path), "--input", str(input_path)]) == 0

    status_output = Path(db_path)
    assert status_output.exists()


def test_run_backfill_builds_and_applies_range(tmp_path, monkeypatch, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"
    output_path = tmp_path / "cycles.json"

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_inputs_from_signal_noise",
        lambda **_kwargs: [
            EvaluationInput(
                date="2026-03-27",
                hypothesis_id="momentum_1d",
                prediction=0.05,
                observation=-0.02,
            ),
            EvaluationInput(
                date="2026-03-28",
                hypothesis_id="momentum_1d",
                prediction=-0.02,
                observation=0.03,
            ),
        ],
    )
    _register_hypothesis(main, db_path, "momentum_1d")
    capsys.readouterr()

    rc = main(
        [
            "apply-backfill",
            "--db",
            str(db_path),
            "--start-date",
            "2026-03-27",
            "--end-date",
            "2026-03-28",
            "--hypothesis-id",
            "momentum_1d",
            "--out",
            str(output_path),
        ]
    )
    assert rc == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 2

    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT input_source, input_range_start, input_range_end, signal_name
            FROM evaluation_snapshots
            ORDER BY evaluation_id
            """
        ).fetchall()
        assert rows == [
            ("signal_noise_backfill", "2026-03-27", "2026-03-28", "btc_ohlcv"),
            ("signal_noise_backfill", "2026-03-27", "2026-03-28", "btc_ohlcv"),
        ]
    finally:
        conn.close()

    output = capsys.readouterr().out
    assert "Wrote evaluation inputs:" in output
    assert "Batch complete: evaluations=2 created=2 existing=0" in output


def test_apply_hypotheses_backfill_applies_multiple_hypotheses(tmp_path, monkeypatch, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"

    def fake_generate_evaluation_inputs_from_signal_noise(**kwargs):
        hypothesis_id = kwargs["hypothesis_id"]
        if hypothesis_id == "momentum_1d":
            return [
                EvaluationInput(
                    date="2026-03-27",
                    hypothesis_id="momentum_1d",
                    prediction=0.05,
                    observation=-0.02,
                ),
                EvaluationInput(
                    date="2026-03-28",
                    hypothesis_id="momentum_1d",
                    prediction=-0.02,
                    observation=0.03,
                ),
            ]
        if hypothesis_id == "reversal_1d":
            return [
                EvaluationInput(
                    date="2026-03-27",
                    hypothesis_id="reversal_1d",
                    prediction=-0.05,
                    observation=-0.02,
                ),
                EvaluationInput(
                    date="2026-03-28",
                    hypothesis_id="reversal_1d",
                    prediction=0.02,
                    observation=0.03,
                ),
            ]
        raise AssertionError(f"unexpected hypothesis: {hypothesis_id}")

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_inputs_from_signal_noise",
        fake_generate_evaluation_inputs_from_signal_noise,
    )
    _register_hypothesis(main, db_path, "momentum_1d")
    _register_hypothesis(main, db_path, "reversal_1d")
    capsys.readouterr()

    assert (
        main(
            [
                "apply-hypotheses-backfill",
                "--db",
                str(db_path),
                "--start-date",
                "2026-03-27",
                "--end-date",
                "2026-03-28",
                "--hypothesis-id",
                "momentum_1d",
                "--hypothesis-id",
                "reversal_1d",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Batch complete: hypotheses=2 evaluations=4 created=4 existing=0" in output
    assert "alpha-os hypothesis competition" in output
    assert "momentum_1d " in output
    assert "reversal_1d " in output
    assert "corr=" in output
    assert "mmc=" in output
    assert "evals=2" in output

    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        counts = {
            "predictions": conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            "observations": conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
            "snapshots": conn.execute("SELECT COUNT(*) FROM evaluation_snapshots").fetchone()[0],
            "metrics": conn.execute("SELECT COUNT(*) FROM hypothesis_metrics").fetchone()[0],
        }
    finally:
        conn.close()

    assert counts == {
        "predictions": 4,
        "observations": 2,
        "snapshots": 4,
        "metrics": 2,
    }


def test_show_evaluations_prints_provenance(tmp_path, monkeypatch, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_inputs_from_signal_noise",
        lambda **_kwargs: [
            EvaluationInput(
                date="2026-03-27",
                hypothesis_id="momentum_1d",
                prediction=0.05,
                observation=-0.02,
            ),
            EvaluationInput(
                date="2026-03-28",
                hypothesis_id="momentum_1d",
                prediction=-0.02,
                observation=0.03,
            ),
        ],
    )
    _register_hypothesis(main, db_path, "momentum_1d")
    capsys.readouterr()

    assert (
        main(
            [
            "apply-backfill",
                "--db",
                str(db_path),
                "--start-date",
                "2026-03-27",
                "--end-date",
                "2026-03-28",
                "--hypothesis-id",
                "momentum_1d",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["show-evaluations", "--db", str(db_path), "--limit", "2"]) == 0
    output = capsys.readouterr().out
    assert "alpha-os evaluations" in output
    assert "source=signal_noise_backfill" in output
    assert "signal=btc_ohlcv" in output
    assert "range=2026-03-27->2026-03-28" in output


def test_v1_smoke_flow_builds_applies_and_audits(tmp_path, monkeypatch, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"
    input_path = tmp_path / "cycles.json"

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_inputs_from_signal_noise",
        lambda **_kwargs: [
            EvaluationInput(
                date="2026-03-27",
                hypothesis_id="momentum_1d",
                prediction=0.05,
                observation=-0.02,
            ),
            EvaluationInput(
                date="2026-03-28",
                hypothesis_id="momentum_1d",
                prediction=-0.02,
                observation=0.03,
            ),
        ],
    )
    _register_hypothesis(main, db_path, "momentum_1d")
    capsys.readouterr()

    assert (
        main(
            [
                "generate-evaluation-inputs",
                "--db",
                str(db_path),
                "--start-date",
                "2026-03-27",
                "--end-date",
                "2026-03-28",
                "--hypothesis-id",
                "momentum_1d",
                "--out",
                str(input_path),
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert "Generated evaluation inputs:" in build_output
    assert "Count:    2" in build_output
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    assert [item["date"] for item in payload] == ["2026-03-27", "2026-03-28"]

    assert (
        main(
            [
                "apply-backfill",
                "--db",
                str(db_path),
                "--start-date",
                "2026-03-27",
                "--end-date",
                "2026-03-28",
                "--hypothesis-id",
                "momentum_1d",
            ]
        )
        == 0
    )
    backfill_output = capsys.readouterr().out
    assert "Batch complete: evaluations=2 created=2 existing=0" in backfill_output

    assert main(["status", "--db", str(db_path)]) == 0
    status_output = capsys.readouterr().out
    assert "alpha-os status" in status_output
    assert "Latest:   BTC:residual_return_1d:2026-03-28 / momentum_1d" in status_output
    assert "Metrics:  tracked=1" in status_output

    assert main(["show-evaluations", "--db", str(db_path), "--limit", "5"]) == 0
    cycles_output = capsys.readouterr().out
    assert "alpha-os evaluations" in cycles_output
    assert "Count:    2" in cycles_output
    assert "source=signal_noise_backfill" in cycles_output
    assert "range=2026-03-27->2026-03-28" in cycles_output


def test_generate_evaluation_input_requires_existing_hypothesis(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    output_path = tmp_path / "cycle.json"

    try:
        main(
            [
                "generate-evaluation-input",
                "--db",
                str(db_path),
                "--date",
                "2026-03-27",
                "--hypothesis-id",
                "momentum_1d",
                "--out",
                str(output_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for unknown hypothesis generation")


def test_generate_evaluation_input_uses_active_definition_from_db(tmp_path, monkeypatch):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    output_path = tmp_path / "cycle.json"
    _register_hypothesis(main, db_path, "momentum_1d")

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.conn.execute(
            """
            UPDATE hypotheses
            SET definition_json = ?
            WHERE hypothesis_id = 'momentum_1d'
            """,
            (
                json.dumps(
                    {
                        "kind": "momentum",
                        "signal_name": "btc_ohlcv",
                        "params": {"lookback": 3},
                    },
                    sort_keys=True,
                ),
            ),
        )
        store.conn.commit()
    finally:
        store.close()

    class FakeClient:
        def get_data(self, name: str, since: str | None = None, resolution: str | None = None):
            return pd.DataFrame(
                [
                    {"timestamp": "2026-03-24T00:00:00+00:00", "value": 100.0},
                    {"timestamp": "2026-03-25T00:00:00+00:00", "value": 110.0},
                    {"timestamp": "2026-03-26T00:00:00+00:00", "value": 121.0},
                    {"timestamp": "2026-03-27T00:00:00+00:00", "value": 133.1},
                    {"timestamp": "2026-03-28T00:00:00+00:00", "value": 146.41},
                ]
            )

    monkeypatch.setattr(
        "alpha_os.evaluation_generation.build_signal_client",
        lambda **_kwargs: FakeClient(),
    )

    assert (
        main(
            [
                "generate-evaluation-input",
                "--db",
                str(db_path),
                "--date",
                "2026-03-27",
                "--hypothesis-id",
                "momentum_1d",
                "--out",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["hypothesis_id"] == "momentum_1d"
    assert payload["prediction"] == pytest.approx(0.1)
    assert payload["observation"] == pytest.approx(0.1)
