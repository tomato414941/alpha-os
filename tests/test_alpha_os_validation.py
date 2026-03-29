from __future__ import annotations

import json


def test_write_and_load_default_validation_spec(tmp_path):
    from alpha_os.validation_spec import (
        default_validation_spec,
        load_validation_spec,
        write_validation_spec,
    )

    path = tmp_path / "validation_spec.json"
    expected = default_validation_spec()
    write_validation_spec(path, expected)
    loaded = load_validation_spec(path)

    assert loaded == expected


def test_run_validation_persists_results(tmp_path):
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec
    import alpha_os.validation_service as validation_service

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()

    frame = {
        "timestamp": [
            "2026-03-20T00:00:00Z",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            "2026-03-23T00:00:00Z",
            "2026-03-24T00:00:00Z",
            "2026-03-25T00:00:00Z",
        ],
        "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
    }

    def _fake_loader(*, base_url: str, signal_name: str):
        import pandas as pd

        return pd.DataFrame(frame)

    original_loader = validation_service._load_price_frame_from_signal_noise
    validation_service._load_price_frame_from_signal_noise = _fake_loader
    try:
        spec = ValidationSpec(
            hypothesis_ids=("momentum_1d", "reversal_1d"),
            target_ids=("residual_return_1d", "residual_return_3d"),
            date_ranges=(
                ValidationDateRange(
                    label="mini",
                    start_date="2026-03-22",
                    end_date="2026-03-22",
                ),
            ),
            metric_windows=(20,),
            aggregation_kinds=("active_equal_mean", "corr_weighted_mean"),
            base_url="http://example.com",
        )
        result = run_validation(store, spec=spec, recorded_at="2026-03-29T00:00:00+00:00")
    finally:
        validation_service._load_price_frame_from_signal_noise = original_loader
        store.close()

    store = EvaluationStore(db_path)
    try:
        run = store.get_validation_run(result.run_id)
        assert run is not None
        assert json.loads(run.spec_json)["target_ids"] == [
            "residual_return_1d",
            "residual_return_3d",
        ]
        hypothesis_results = store.list_validation_hypothesis_results(run_id=result.run_id)
        meta_results = store.list_validation_meta_results(run_id=result.run_id)
        assert len(hypothesis_results) == 4
        assert len(meta_results) == 4
        assert {item.target_id for item in hypothesis_results} == {
            "residual_return_1d",
            "residual_return_3d",
        }
        assert {item.aggregation_kind for item in meta_results} == {
            "active_equal_mean",
            "corr_weighted_mean",
        }
    finally:
        store.close()


def test_run_validation_clips_unavailable_tail_dates(tmp_path):
    from alpha_os.store import EvaluationStore
    from alpha_os.validation_service import run_validation
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec
    import alpha_os.validation_service as validation_service

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    store.ensure_schema()

    frame = {
        "timestamp": [
            "2026-03-20T00:00:00Z",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            "2026-03-23T00:00:00Z",
            "2026-03-24T00:00:00Z",
            "2026-03-25T00:00:00Z",
        ],
        "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
    }

    def _fake_loader(*, base_url: str, signal_name: str):
        import pandas as pd

        return pd.DataFrame(frame)

    original_loader = validation_service._load_price_frame_from_signal_noise
    validation_service._load_price_frame_from_signal_noise = _fake_loader
    try:
        spec = ValidationSpec(
            hypothesis_ids=("momentum_1d",),
            target_ids=("residual_return_3d",),
            date_ranges=(
                ValidationDateRange(
                    label="tail_clipped",
                    start_date="2026-03-20",
                    end_date="2026-03-25",
                ),
            ),
            metric_windows=(20,),
            aggregation_kinds=("active_equal_mean",),
            base_url="http://example.com",
        )
        result = run_validation(store, spec=spec, recorded_at="2026-03-29T00:00:00+00:00")
    finally:
        validation_service._load_price_frame_from_signal_noise = original_loader
        store.close()

    store = EvaluationStore(db_path)
    try:
        hypothesis_results = store.list_validation_hypothesis_results(run_id=result.run_id)
        assert len(hypothesis_results) == 1
        assert hypothesis_results[0].sample_count == 2
    finally:
        store.close()


def test_validation_cli_roundtrip(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec, write_validation_spec

    db_path = tmp_path / "runtime.db"
    spec_path = tmp_path / "validation_spec.json"
    spec = ValidationSpec(
        hypothesis_ids=("momentum_1d", "reversal_1d"),
        target_ids=("residual_return_1d",),
        date_ranges=(
            ValidationDateRange(
                label="mini",
                start_date="2026-03-20",
                end_date="2026-03-24",
            ),
        ),
        metric_windows=(20,),
        aggregation_kinds=("active_equal_mean",),
        base_url="https://signal-noise.taildd87b4.ts.net",
    )
    write_validation_spec(spec_path, spec)

    assert main(["run-validation", "--db", str(db_path), "--spec", str(spec_path)]) == 0
    run_output = capsys.readouterr().out
    assert "Validation complete" in run_output

    assert main(["show-validation", "--db", str(db_path)]) == 0
    show_output = capsys.readouterr().out
    assert "alpha-os validation" in show_output
    assert "Hypothesis Results:" in show_output
    assert "Meta Results:" in show_output

    assert main(["summarize-validation", "--db", str(db_path)]) == 0
    summary_output = capsys.readouterr().out
    assert "alpha-os validation summary" in summary_output
    assert "Meta Aggregations:" in summary_output
