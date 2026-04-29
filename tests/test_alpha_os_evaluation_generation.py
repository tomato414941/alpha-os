from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _register_signal(main, db_path, signal_id: str) -> None:
    assert (
        main(
            [
                "register-signal-candidate",
                "--db",
                str(db_path),
                "--signal-candidate-id",
                signal_id,
            ]
        )
        == 0
    )


def test_generate_evaluation_input_from_frame_uses_prev_and_next_close():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 146.41},
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 161.051},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 177.1561},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="momentum_1d",
    )

    assert evaluation_input.date == "2026-03-27"
    assert evaluation_input.signal_id == "momentum_1d"
    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx((177.1561 / 133.1) - 1.0)


def test_generate_evaluation_input_rejects_missing_neighbor_rows():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
        ]
    )

    try:
        generate_evaluation_input_from_frame(
            frame=frame,
            date="2026-03-27",
            signal_id="momentum_1d",
        )
    except ValueError as exc:
        assert "future close 3 days ahead" in str(exc)
    else:
        raise AssertionError("expected previous-close validation error")


def test_generate_evaluation_inputs_from_frame_uses_date_range():
    from alpha_os.evaluation_generation import generate_evaluation_inputs_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 105.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 115.5},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 121.275},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 127.33875},
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 133.7056875},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 140.390971875},
            {"timestamp": "2026-03-31T00:00:00+00:00", "close": 147.41052046875},
        ]
    )

    evaluation_inputs = generate_evaluation_inputs_from_frame(
        frame=frame,
        start_date="2026-03-26",
        end_date="2026-03-28",
        signal_id="momentum_1d",
    )

    assert [item.date for item in evaluation_inputs] == [
        "2026-03-26",
        "2026-03-27",
        "2026-03-28",
    ]
    assert all(item.signal_id == "momentum_1d" for item in evaluation_inputs)
    assert evaluation_inputs[0].observation == pytest.approx((133.7056875 / 115.5) - 1.0)


def test_generate_evaluation_inputs_batch_from_feature_plane_matches_single_generation():
    from alpha_os.signal_compiler import compile_signal_families
    from alpha_os.evaluation_generation import (
        generate_evaluation_inputs_batch_from_feature_plane,
        generate_evaluation_inputs_from_feature_plane,
        prepare_feature_plane_from_frame,
    )
    from alpha_os.signal_registry import get_signal_definition

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 105.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 115.5},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 121.275},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 127.33875},
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 133.7056875},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 140.390971875},
            {"timestamp": "2026-03-31T00:00:00+00:00", "close": 147.41052046875},
        ]
    )

    plane = prepare_feature_plane_from_frame(frame=frame)
    definitions = [
        get_signal_definition("momentum_1d"),
        get_signal_definition("reversal_1d"),
        get_signal_definition("average_gap_3d"),
    ]
    compiled = compile_signal_families(definitions)
    assert len(compiled) == 3
    assert {family.kind for family in compiled} == {
        "momentum",
        "reversal",
        "average_gap",
    }

    batch_items = generate_evaluation_inputs_batch_from_feature_plane(
        plane=plane,
        start_date="2026-03-26",
        end_date="2026-03-28",
        definitions=definitions,
    )
    single_items = []
    for definition in definitions:
        single_items.extend(
            generate_evaluation_inputs_from_feature_plane(
                plane=plane,
                start_date="2026-03-26",
                end_date="2026-03-28",
                signal_id=definition.signal_id,
                definition=definition,
            )
        )

    assert sorted(
        (item.signal_id, item.date, item.prediction, item.observation)
        for item in batch_items
    ) == sorted(
        (item.signal_id, item.date, item.prediction, item.observation)
        for item in single_items
    )


def test_generate_evaluation_input_from_frame_supports_momentum_3d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 146.41},
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 161.051},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 177.1561},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="momentum_3d",
    )

    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx((177.1561 / 133.1) - 1.0)


def test_generate_evaluation_input_from_frame_supports_vol_compression_breakout():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    rows = []
    close = 100.0
    for i, timestamp in enumerate(pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")):
        if i < 10:
            close *= 1.03
        elif i < 30:
            close *= 1.002
        else:
            close *= 1.001
        rows.append({"timestamp": timestamp.isoformat(), "close": close})
    frame = pd.DataFrame(rows)

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-04-20",
        signal_id="vol_compression_breakout_20d",
    )

    assert evaluation_input.prediction == pytest.approx(0.001)
    assert evaluation_input.observation == pytest.approx(0.0030030009999995055)


def test_generate_evaluation_input_from_frame_supports_trend_volume_confirmation():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    rows = []
    close = 100.0
    for i, timestamp in enumerate(pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")):
        if i < 10:
            close *= 1.03
        elif i < 30:
            close *= 1.002
        else:
            close *= 1.001
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "close": close,
                "volume": 10_000.0 + 500.0 * i,
            }
        )
    frame = pd.DataFrame(rows)

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-04-20",
        signal_id="trend_volume_confirmation_20d",
    )

    assert evaluation_input.prediction == pytest.approx(0.0011990850278772263)
    assert evaluation_input.observation == pytest.approx(0.0030030009999995055)


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
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 194.87171},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 214.358881},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="momentum_5d",
    )

    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx((214.358881 / 161.051) - 1.0)


def test_generate_evaluation_input_from_frame_supports_reversal_1d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 146.41},
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 161.051},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 177.1561},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="reversal_1d",
    )

    assert evaluation_input.prediction == pytest.approx(-0.1)
    assert evaluation_input.observation == pytest.approx((177.1561 / 133.1) - 1.0)


def test_generate_evaluation_input_from_frame_supports_reversal_3d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 146.41},
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 161.051},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 177.1561},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="reversal_3d",
    )

    assert evaluation_input.prediction == pytest.approx(-0.1)
    assert evaluation_input.observation == pytest.approx((177.1561 / 133.1) - 1.0)


def test_generate_evaluation_input_from_frame_supports_average_gap_3d():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {"timestamp": "2026-03-24T00:00:00+00:00", "close": 100.0},
            {"timestamp": "2026-03-25T00:00:00+00:00", "close": 110.0},
            {"timestamp": "2026-03-26T00:00:00+00:00", "close": 121.0},
            {"timestamp": "2026-03-27T00:00:00+00:00", "close": 133.1},
            {"timestamp": "2026-03-28T00:00:00+00:00", "close": 146.41},
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 161.051},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 177.1561},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="average_gap_3d",
    )

    expected = (133.1 / ((110.0 + 121.0 + 133.1) / 3.0)) - 1.0
    assert evaluation_input.prediction == pytest.approx(expected)
    assert evaluation_input.observation == pytest.approx((177.1561 / 133.1) - 1.0)


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
            {"timestamp": "2026-03-29T00:00:00+00:00", "close": 112.0},
            {"timestamp": "2026-03-30T00:00:00+00:00", "close": 113.0},
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="range_position_5d",
    )

    window = [104.0, 102.0, 108.0, 106.0, 110.0]
    expected = ((110.0 - min(window)) / (max(window) - min(window))) * 2.0 - 1.0
    assert evaluation_input.prediction == pytest.approx(expected)
    assert evaluation_input.observation == pytest.approx((113.0 / 110.0) - 1.0)


def test_generate_evaluation_input_from_signal_noise_uses_value_series(monkeypatch):
    from alpha_os.evaluation_generation import generate_evaluation_input_from_signal_noise

    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def get_observation_data(
            self,
            *,
            asset: str,
            observable_id: str,
            since: str | None = None,
            resolution: str = "1d",
            source_id: str = "signal_noise",
        ):
            calls.append((asset, observable_id, resolution))
            return pd.DataFrame(
                [
                    {"timestamp": "2026-03-24T00:00:00+00:00", "value": 100.0},
                    {"timestamp": "2026-03-25T00:00:00+00:00", "value": 110.0},
                    {"timestamp": "2026-03-26T00:00:00+00:00", "value": 121.0},
                    {"timestamp": "2026-03-27T00:00:00+00:00", "value": 133.1},
                    {"timestamp": "2026-03-28T00:00:00+00:00", "value": 146.41},
                    {"timestamp": "2026-03-29T00:00:00+00:00", "value": 161.051},
                    {"timestamp": "2026-03-30T00:00:00+00:00", "value": 177.1561},
                ]
            )

    monkeypatch.setattr(
        "alpha_os.observation_adapters.build_signal_client",
        lambda **_kwargs: FakeClient(),
    )

    evaluation_input = generate_evaluation_input_from_signal_noise(
        date="2026-03-27",
        signal_id="momentum_1d",
        base_url="https://signal-noise.example",
    )

    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx((177.1561 / 133.1) - 1.0)
    assert calls == [("BTC", "daily_close", "1d")]


def test_cmd_generate_evaluation_input_writes_json(tmp_path, monkeypatch, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"
    output_path = tmp_path / "cycle.json"

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_input_from_signal_noise",
        lambda **_kwargs: EvaluationInput(
            date="2026-03-27",
            signal_id="momentum_1d",
            prediction=0.05,
            observation=-0.02,
        ),
    )
    _register_signal(main, db_path, "momentum_1d")
    capsys.readouterr()

    rc = main(
        [
            "debug-generate-evaluation-input",
            "--db",
            str(db_path),
            "--date",
            "2026-03-27",
            "--signal-candidate-id",
            "momentum_1d",
            "--out",
            str(output_path),
        ]
    )
    assert rc == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-03-27"
    assert payload["signal_id"] == "momentum_1d"
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
                signal_id="momentum_1d",
                prediction=0.05,
                observation=-0.02,
            ),
            EvaluationInput(
                date="2026-03-28",
                signal_id="momentum_1d",
                prediction=-0.02,
                observation=0.03,
            ),
        ],
    )
    _register_signal(main, db_path, "momentum_1d")
    capsys.readouterr()

    rc = main(
        [
            "debug-generate-evaluation-inputs",
            "--db",
            str(db_path),
            "--start-date",
            "2026-03-27",
            "--end-date",
            "2026-03-28",
            "--signal-candidate-id",
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
            signal_id="momentum_1d",
            prediction=0.05,
            observation=-0.02,
        ),
    )
    _register_signal(main, db_path, "momentum_1d")

    assert (
        main(
            [
                "debug-generate-evaluation-input",
                "--db",
                str(db_path),
                "--date",
                "2026-03-27",
                "--signal-candidate-id",
                "momentum_1d",
                "--out",
                str(input_path),
            ]
        )
        == 0
    )
    assert main(["debug-apply-evaluation", "--db", str(db_path), "--input", str(input_path)]) == 0

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
                signal_id="momentum_1d",
                prediction=0.05,
                observation=-0.02,
            ),
            EvaluationInput(
                date="2026-03-28",
                signal_id="momentum_1d",
                prediction=-0.02,
                observation=0.03,
            ),
        ],
    )
    _register_signal(main, db_path, "momentum_1d")
    capsys.readouterr()

    rc = main(
        [
            "debug-apply-backfill",
            "--db",
            str(db_path),
            "--start-date",
            "2026-03-27",
            "--end-date",
            "2026-03-28",
            "--signal-candidate-id",
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
            SELECT input_source, input_range_start, input_range_end,
                   observation_spec_id, observable_id, adapter_kind
            FROM evaluation_snapshots
            ORDER BY evaluation_id
            """
        ).fetchall()
        assert rows == [
            (
                "signal_noise_backfill",
                "2026-03-27",
                "2026-03-28",
                "momentum_1d__default",
                "daily_close",
                "signal_noise_asset_observable",
            ),
            (
                "signal_noise_backfill",
                "2026-03-27",
                "2026-03-28",
                "momentum_1d__default",
                "daily_close",
                "signal_noise_asset_observable",
            ),
        ]
    finally:
        conn.close()

    output = capsys.readouterr().out
    assert "Wrote evaluation inputs:" in output
    assert "Batch complete: evaluations=2 created=2 existing=0" in output


def test_apply_signals_backfill_applies_multiple_signals(tmp_path, monkeypatch, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_inputs import EvaluationInput

    db_path = tmp_path / "runtime.db"

    def fake_generate_evaluation_inputs_from_signal_noise(**kwargs):
        signal_id = kwargs["signal_id"]
        if signal_id == "momentum_1d":
            return [
                EvaluationInput(
                    date="2026-03-27",
                    signal_id="momentum_1d",
                    prediction=0.05,
                    observation=-0.02,
                ),
                EvaluationInput(
                    date="2026-03-28",
                    signal_id="momentum_1d",
                    prediction=-0.02,
                    observation=0.03,
                ),
            ]
        if signal_id == "reversal_1d":
            return [
                EvaluationInput(
                    date="2026-03-27",
                    signal_id="reversal_1d",
                    prediction=-0.05,
                    observation=-0.02,
                ),
                EvaluationInput(
                    date="2026-03-28",
                    signal_id="reversal_1d",
                    prediction=0.02,
                    observation=0.03,
                ),
            ]
        raise AssertionError(f"unexpected signal: {signal_id}")

    monkeypatch.setattr(
        "alpha_os.cli.generate_evaluation_inputs_from_signal_noise",
        fake_generate_evaluation_inputs_from_signal_noise,
    )
    _register_signal(main, db_path, "momentum_1d")
    _register_signal(main, db_path, "reversal_1d")
    capsys.readouterr()

    assert (
        main(
            [
                "debug-apply-signal-candidates-backfill",
                "--db",
                str(db_path),
                "--start-date",
                "2026-03-27",
                "--end-date",
                "2026-03-28",
                "--signal-candidate-id",
                "momentum_1d",
                "--signal-candidate-id",
                "reversal_1d",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Batch complete: signals=2 evaluations=4 created=4 existing=0" in output
    assert "alpha-os signal competition" in output
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
            "metrics": conn.execute("SELECT COUNT(*) FROM signal_metrics").fetchone()[0],
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
                signal_id="momentum_1d",
                prediction=0.05,
                observation=-0.02,
            ),
            EvaluationInput(
                date="2026-03-28",
                signal_id="momentum_1d",
                prediction=-0.02,
                observation=0.03,
            ),
        ],
    )
    _register_signal(main, db_path, "momentum_1d")
    capsys.readouterr()

    assert (
        main(
            [
            "debug-apply-backfill",
                "--db",
                str(db_path),
                "--start-date",
                "2026-03-27",
                "--end-date",
                "2026-03-28",
                "--signal-candidate-id",
                "momentum_1d",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["debug-show-evaluations", "--db", str(db_path), "--limit", "2"]) == 0
    output = capsys.readouterr().out
    assert "alpha-os evaluations" in output
    assert "source=signal_noise_backfill" in output
    assert "observation=momentum_1d__default=daily_close@signal_noise_asset_observable" in output
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
                signal_id="momentum_1d",
                prediction=0.05,
                observation=-0.02,
            ),
            EvaluationInput(
                date="2026-03-28",
                signal_id="momentum_1d",
                prediction=-0.02,
                observation=0.03,
            ),
        ],
    )
    _register_signal(main, db_path, "momentum_1d")
    capsys.readouterr()

    assert (
        main(
            [
                "debug-generate-evaluation-inputs",
                "--db",
                str(db_path),
                "--start-date",
                "2026-03-27",
                "--end-date",
                "2026-03-28",
                "--signal-candidate-id",
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
                "debug-apply-backfill",
                "--db",
                str(db_path),
                "--start-date",
                "2026-03-27",
                "--end-date",
                "2026-03-28",
                "--signal-candidate-id",
                "momentum_1d",
            ]
        )
        == 0
    )
    backfill_output = capsys.readouterr().out
    assert "Batch complete: evaluations=2 created=2 existing=0" in backfill_output

    assert main(["debug-status", "--db", str(db_path)]) == 0
    status_output = capsys.readouterr().out
    assert "alpha-os status" in status_output
    assert "Latest:   BTC:residual_return_3d:2026-03-28 / momentum_1d" in status_output
    assert "Metrics:  tracked=1" in status_output

    assert (
        main(["debug-show-evaluations", "--db", str(db_path), "--limit", "5"]) == 0
    )
    cycles_output = capsys.readouterr().out
    assert "alpha-os evaluations" in cycles_output
    assert "Count:    2" in cycles_output
    assert "source=signal_noise_backfill" in cycles_output
    assert "range=2026-03-27->2026-03-28" in cycles_output


def test_generate_evaluation_input_requires_existing_signal(tmp_path):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    output_path = tmp_path / "cycle.json"

    try:
        main(
            [
                "debug-generate-evaluation-input",
                "--db",
                str(db_path),
                "--date",
                "2026-03-27",
                "--signal-candidate-id",
                "momentum_1d",
                "--out",
                str(output_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parser exit for unknown signal generation")


def test_generate_evaluation_input_uses_active_definition_from_db(tmp_path, monkeypatch):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    output_path = tmp_path / "cycle.json"
    _register_signal(main, db_path, "momentum_1d")

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.conn.execute(
            """
            UPDATE signals
            SET definition_json = ?
            WHERE signal_id = 'momentum_1d'
            """,
                (
                    json.dumps(
                        {
                            "kind": "momentum",
                            "signal_name": "btc_ohlcv",
                            "target_definition": {
                                "target_id": "residual_return_3d",
                                "family": "residual_return",
                                "observation_kind": "fixed_horizon",
                                "subject_kind": "asset",
                                "output_kind": "real_value",
                                "scoring_kind": "corr_mmc",
                                "params": {"horizon_days": 3},
                            },
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
        def get_observation_data(
            self,
            *,
            asset: str,
            observable_id: str,
            since: str | None = None,
            resolution: str = "1d",
            source_id: str = "signal_noise",
        ):
            return pd.DataFrame(
                [
                    {"timestamp": "2026-03-24T00:00:00+00:00", "value": 100.0},
                    {"timestamp": "2026-03-25T00:00:00+00:00", "value": 110.0},
                    {"timestamp": "2026-03-26T00:00:00+00:00", "value": 121.0},
                    {"timestamp": "2026-03-27T00:00:00+00:00", "value": 133.1},
                    {"timestamp": "2026-03-28T00:00:00+00:00", "value": 146.41},
                    {"timestamp": "2026-03-29T00:00:00+00:00", "value": 161.051},
                    {"timestamp": "2026-03-30T00:00:00+00:00", "value": 177.1561},
                ]
            )

    monkeypatch.setattr(
        "alpha_os.observation_adapters.build_signal_client",
        lambda **_kwargs: FakeClient(),
    )

    assert (
        main(
            [
                "debug-generate-evaluation-input",
                "--db",
                str(db_path),
                "--date",
                "2026-03-27",
                "--signal-candidate-id",
                "momentum_1d",
                "--out",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["signal_id"] == "momentum_1d"
    assert payload["prediction"] == pytest.approx(0.1)
    assert payload["observation"] == pytest.approx((177.1561 / 133.1) - 1.0)


def test_generate_evaluation_input_from_frame_emits_instrument_artifacts():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-24T00:00:00+00:00",
                "close": 100.0,
                "funding_rate": 0.0001,
                "borrow_fee": 0.0002,
                "roll_cost_bps": 3.5,
            },
            {
                "timestamp": "2026-03-25T00:00:00+00:00",
                "close": 110.0,
                "funding_rate": 0.0002,
                "borrow_fee": 0.0003,
                "roll_cost_bps": 4.5,
            },
            {
                "timestamp": "2026-03-26T00:00:00+00:00",
                "close": 121.0,
                "funding_rate": 0.0003,
                "borrow_fee": 0.0004,
                "roll_cost_bps": 5.5,
            },
            {
                "timestamp": "2026-03-27T00:00:00+00:00",
                "close": 133.1,
                "funding_rate": 0.0004,
                "borrow_fee": 0.0005,
                "roll_cost_bps": 6.5,
            },
            {
                "timestamp": "2026-03-28T00:00:00+00:00",
                "close": 146.41,
                "funding_rate": 0.0005,
                "borrow_fee": 0.0006,
                "roll_cost_bps": 7.5,
            },
            {
                "timestamp": "2026-03-29T00:00:00+00:00",
                "close": 161.051,
                "funding_rate": 0.0006,
                "borrow_fee": 0.0007,
                "roll_cost_bps": 8.5,
            },
            {
                "timestamp": "2026-03-30T00:00:00+00:00",
                "close": 177.1561,
                "funding_rate": 0.0007,
                "borrow_fee": 0.0008,
                "roll_cost_bps": 9.5,
            },
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="momentum_1d",
        contract_multiplier=5.0,
    )

    assert evaluation_input.funding_cost_bps == pytest.approx(4.0)
    assert evaluation_input.borrow_fee_bps == pytest.approx(5.0)
    assert evaluation_input.roll_cost_bps == pytest.approx(6.5)
    assert evaluation_input.contract_multiplier == pytest.approx(5.0)


def test_generate_evaluation_input_from_frame_separates_research_and_tradable_prices():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame
    from alpha_os.portfolio_decision import ObservationSpec

    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-24T00:00:00+00:00",
                "research_close": 100.0,
                "tradable_price": 120.0,
            },
            {
                "timestamp": "2026-03-25T00:00:00+00:00",
                "research_close": 110.0,
                "tradable_price": 126.0,
            },
            {
                "timestamp": "2026-03-26T00:00:00+00:00",
                "research_close": 121.0,
                "tradable_price": 132.3,
            },
            {
                "timestamp": "2026-03-27T00:00:00+00:00",
                "research_close": 133.1,
                "tradable_price": 138.915,
            },
            {
                "timestamp": "2026-03-28T00:00:00+00:00",
                "research_close": 146.41,
                "tradable_price": 145.86075,
            },
            {
                "timestamp": "2026-03-29T00:00:00+00:00",
                "research_close": 161.051,
                "tradable_price": 153.1537875,
            },
            {
                "timestamp": "2026-03-30T00:00:00+00:00",
                "research_close": 177.1561,
                "tradable_price": 160.811476875,
            },
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="momentum_1d",
        observation_spec=ObservationSpec(
            observation_spec_id="futures_price_split",
            adapter_kind="signal_noise_asset_observable",
            observable_id="daily_close",
            research_price_observable_id="daily_close",
            tradable_price_observable_id="tradable_price",
        ),
    )

    assert evaluation_input.prediction == pytest.approx(0.1)
    assert evaluation_input.observation == pytest.approx(
        (160.811476875 / 138.915) - 1.0
    )


def test_generate_evaluation_input_from_frame_emits_lifecycle_metadata():
    from alpha_os.evaluation_generation import generate_evaluation_input_from_frame

    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-24T00:00:00+00:00",
                "close": 100.0,
                "tradable_price": 100.0,
                "contract_id": "ESM2026",
                "next_contract_id": "ESU2026",
                "expiry": "2026-03-30",
                "contract_family": "CME:ES",
                "quote_ccy": "USD",
                "collateral_ccy": "USD",
                "financing_cost_bps": 0.0002,
            },
            {
                "timestamp": "2026-03-25T00:00:00+00:00",
                "close": 110.0,
                "tradable_price": 110.0,
                "contract_id": "ESM2026",
                "next_contract_id": "ESU2026",
                "expiry": "2026-03-30",
                "contract_family": "CME:ES",
                "quote_ccy": "USD",
                "collateral_ccy": "USD",
                "financing_cost_bps": 0.0002,
            },
            {
                "timestamp": "2026-03-26T00:00:00+00:00",
                "close": 121.0,
                "tradable_price": 121.0,
                "contract_id": "ESM2026",
                "next_contract_id": "ESU2026",
                "expiry": "2026-03-30",
                "contract_family": "CME:ES",
                "quote_ccy": "USD",
                "collateral_ccy": "USD",
                "financing_cost_bps": 0.0002,
            },
            {
                "timestamp": "2026-03-27T00:00:00+00:00",
                "close": 133.1,
                "tradable_price": 133.1,
                "contract_id": "ESM2026",
                "next_contract_id": "ESU2026",
                "expiry": "2026-03-30",
                "contract_family": "CME:ES",
                "quote_ccy": "USD",
                "collateral_ccy": "USD",
                "financing_cost_bps": 0.0002,
            },
            {
                "timestamp": "2026-03-28T00:00:00+00:00",
                "close": 146.41,
                "tradable_price": 146.41,
                "contract_id": "ESM2026",
                "next_contract_id": "ESU2026",
                "expiry": "2026-03-30",
                "contract_family": "CME:ES",
                "quote_ccy": "USD",
                "collateral_ccy": "USD",
                "financing_cost_bps": 0.0002,
            },
            {
                "timestamp": "2026-03-29T00:00:00+00:00",
                "close": 161.051,
                "tradable_price": 161.051,
                "contract_id": "ESM2026",
                "next_contract_id": "ESU2026",
                "expiry": "2026-03-30",
                "contract_family": "CME:ES",
                "quote_ccy": "USD",
                "collateral_ccy": "USD",
                "financing_cost_bps": 0.0002,
            },
            {
                "timestamp": "2026-03-30T00:00:00+00:00",
                "close": 177.1561,
                "tradable_price": 177.1561,
                "contract_id": "ESM2026",
                "next_contract_id": "ESU2026",
                "expiry": "2026-03-30",
                "contract_family": "CME:ES",
                "quote_ccy": "USD",
                "collateral_ccy": "USD",
                "financing_cost_bps": 0.0002,
            },
        ]
    )

    evaluation_input = generate_evaluation_input_from_frame(
        frame=frame,
        date="2026-03-27",
        signal_id="momentum_1d",
        contract_multiplier=50.0,
        roll_rule="calendar_days_before_expiry:3",
    )

    assert evaluation_input.financing_cost_bps == pytest.approx(2.0)
    assert evaluation_input.contract_multiplier == pytest.approx(50.0)
    assert evaluation_input.contract_id == "ESU2026"
    assert evaluation_input.contract_family == "CME:ES"
    assert evaluation_input.quote_ccy == "USD"
    assert evaluation_input.collateral_ccy == "USD"
    assert evaluation_input.roll_event == {
        "contract_id": "ESU2026",
        "contract_family": "CME:ES",
        "quote_ccy": "USD",
        "collateral_ccy": "USD",
        "expiry": "2026-03-30",
        "days_to_expiry": 3,
        "rolled": True,
        "from_contract_id": "ESM2026",
        "to_contract_id": "ESU2026",
        "roll_reason": "calendar_days_before_expiry",
    }


def test_debug_show_evaluations_includes_replay_artifacts(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("momentum_1d")
        apply_evaluation(
            store,
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            signal_id="momentum_1d",
            prediction_value=0.05,
            observation_value=-0.02,
            funding_cost_bps=1.5,
            borrow_fee_bps=2.5,
            roll_cost_bps=0.75,
            contract_multiplier=3.0,
        )
    finally:
        store.close()

    assert main(["debug-show-evaluations", "--db", str(db_path), "--limit", "5"]) == 0
    output = capsys.readouterr().out

    assert "alpha-os evaluations" in output
    assert "replay=funding_bps=1.500000 borrow_bps=2.500000 roll_bps=0.750000 multiplier=3.000000" in output
