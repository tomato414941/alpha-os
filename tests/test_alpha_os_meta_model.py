from __future__ import annotations

import json
from pathlib import Path
import pytest


def _strategy_portfolio_document(
    *,
    sizing_method: str,
    direction_mode: str | None,
    gross_exposure_cap: float | None,
    selection_kind: str = "all_assets",
    top_k: int | None = None,
    rebalance_interval_steps: int = 1,
    rebalance_friction_policy: dict[str, object] | None = None,
    execution_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    portfolio: dict[str, object] = {
        "portfolio_construction": {
            "sizing_policy": {"sizing_method": sizing_method},
            "direction_mode": direction_mode,
            "gross_exposure_cap": gross_exposure_cap,
        },
        "rebalance_friction_policy": (
            {} if rebalance_friction_policy is None else rebalance_friction_policy
        ),
        "execution_policy": {} if execution_policy is None else execution_policy,
        "rebalance_interval_steps": rebalance_interval_steps,
        "selection_kind": selection_kind,
    }
    if top_k is not None:
        portfolio["top_k"] = top_k
    return {"portfolio": portfolio}


def test_refresh_target_meta_predictions_persists_equal_and_corr_weighted_means(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("hyp_a")
        store.register_signal("hyp_b")

        apply_evaluation(
            store,
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            signal_id="hyp_a",
            prediction_value=0.2,
            observation_value=0.1,
        )
        apply_evaluation(
            store,
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            signal_id="hyp_b",
            prediction_value=0.0,
            observation_value=0.1,
        )

        refresh_target_meta_predictions(store)
        items = store.list_meta_predictions(limit=10)
        by_kind = {item.aggregation_kind: item for item in items}

        assert set(by_kind) == {"active_equal_mean", "corr_weighted_mean"}
        assert by_kind["active_equal_mean"].value == 0.1
        assert by_kind["active_equal_mean"].contributor_count == 2
        assert by_kind["corr_weighted_mean"].contributor_count == 2
        assert by_kind["corr_weighted_mean"].details is not None
    finally:
        store.close()


def test_strategy_adaptation_state_schema_migrates_from_online_learning_table(tmp_path):
    import sqlite3

    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE online_learning_states (
                signal_discovery_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO online_learning_states (
                signal_discovery_id,
                state_json,
                created_at
            ) VALUES (?, ?, ?)
            """,
            (
                "discovery",
                json.dumps(
                    {
                        "signal_discovery_id": "discovery",
                        "source_evaluation_report_id": "report-1",
                        "source_screening_result_id": "discovery:screen",
                        "signal_reputations": [],
                        "family_reputations": [],
                        "created_at": "2026-04-03T00:00:00+00:00",
                    },
                    sort_keys=True,
                ),
                "2026-04-03T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        column_rows = store.conn.execute(
            "PRAGMA table_info(strategy_adaptation_states)"
        ).fetchall()
        primary_keys = {
            str(row["name"]): int(row["pk"])
            for row in column_rows
        }
        assert primary_keys["strategy_id"] == 1
        assert primary_keys["signal_discovery_id"] == 0

        migrated_state = store.get_strategy_adaptation_state("discovery")
        assert migrated_state is not None
        assert migrated_state.strategy_id == "discovery"
        assert migrated_state.signal_discovery_id == "discovery"

        listed_states = store.list_strategy_adaptation_states(
            signal_discovery_id="discovery",
            limit=5,
        )
        assert len(listed_states) == 1
        assert listed_states[0].strategy_id == "discovery"
    finally:
        store.close()


def test_refresh_target_meta_predictions_falls_back_to_equal_mean_when_corr_weights_are_non_positive(
    tmp_path,
):
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("hyp_a")
        store.register_signal("hyp_b")
        store.record_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            signal_id="hyp_a",
            prediction_value=0.2,
        )
        store.record_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-27",
            signal_id="hyp_b",
            prediction_value=0.0,
        )
        refresh_target_meta_predictions(store)
        items = {
            item.aggregation_kind: item
            for item in store.list_meta_predictions(limit=10)
        }
        assert items["active_equal_mean"].value == 0.1
        assert items["corr_weighted_mean"].value == 0.1
    finally:
        store.close()


def test_apply_evaluations_batch_is_idempotent(tmp_path):
    from alpha_os.evaluation_inputs import EvaluationInput
    from alpha_os.evaluation_runtime import apply_evaluations_batch
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("hyp_a")
        store.register_signal("hyp_b")

        evaluation_inputs = [
            EvaluationInput(
                date="2026-03-27",
                signal_id="hyp_a",
                prediction=0.2,
                observation=0.1,
            ),
            EvaluationInput(
                date="2026-03-27",
                signal_id="hyp_b",
                prediction=0.0,
                observation=0.1,
            ),
            EvaluationInput(
                date="2026-03-28",
                signal_id="hyp_a",
                prediction=0.3,
                observation=0.2,
            ),
            EvaluationInput(
                date="2026-03-28",
                signal_id="hyp_b",
                prediction=0.1,
                observation=0.2,
            ),
        ]

        latest_snapshot, created_count, existing_count = apply_evaluations_batch(
            store,
            evaluation_inputs=evaluation_inputs,
            input_source="test_batch",
            refresh_metrics=False,
        )
        assert latest_snapshot is not None
        assert created_count == 4
        assert existing_count == 0

        latest_snapshot, created_count, existing_count = apply_evaluations_batch(
            store,
            evaluation_inputs=evaluation_inputs,
            input_source="test_batch",
            refresh_metrics=False,
        )
        assert latest_snapshot is not None
        assert created_count == 0
        assert existing_count == 4

        assert len(store.list_evaluation_snapshots(limit=10)) == 4
        assert store.get_signal("hyp_a").prediction_count == 2
        assert store.get_signal("hyp_a").observation_count == 2
        assert store.get_signal("hyp_b").prediction_count == 2
        assert store.get_signal("hyp_b").observation_count == 2
    finally:
        store.close()


def test_apply_evaluations_batch_can_skip_meta_prediction_refresh(tmp_path):
    from alpha_os.evaluation_inputs import EvaluationInput
    from alpha_os.evaluation_runtime import apply_evaluations_batch
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("hyp_a")
        store.register_signal("hyp_b")

        _, created_count, _ = apply_evaluations_batch(
            store,
            evaluation_inputs=[
                EvaluationInput(
                    date="2026-03-27",
                    signal_id="hyp_a",
                    prediction=0.2,
                    observation=0.1,
                ),
                EvaluationInput(
                    date="2026-03-27",
                    signal_id="hyp_b",
                    prediction=0.0,
                    observation=0.1,
                ),
            ],
            input_source="test_batch",
            refresh_meta_predictions=False,
        )

        assert created_count == 2
        assert len(store.list_signal_metrics(signal_ids=["hyp_a", "hyp_b"])) == 2
        assert store.list_meta_predictions(limit=10) == []
    finally:
        store.close()


def test_refresh_target_meta_prediction_metrics_persists_corr(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.meta_metrics_service import refresh_target_meta_prediction_metrics
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("hyp_a")
        store.register_signal("hyp_b")

        values = [
            ("2026-03-27", 0.2, 0.0, 0.1),
            ("2026-03-28", 0.4, 0.1, 0.2),
            ("2026-03-29", 0.1, 0.0, 0.05),
        ]
        for date, pred_a, pred_b, obs in values:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="hyp_a",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="hyp_b",
                prediction_value=pred_b,
                observation_value=obs,
            )

        refresh_target_meta_predictions(store)
        refresh_target_meta_prediction_metrics(store)
        metrics = store.list_meta_prediction_metrics()
        by_kind = {item.aggregation_kind: item for item in metrics}

        assert set(by_kind) == {"active_equal_mean", "corr_weighted_mean"}
        assert by_kind["active_equal_mean"].sample_count == 3
        assert by_kind["corr_weighted_mean"].sample_count == 3
    finally:
        store.close()


def test_refresh_target_meta_predictions_uses_only_lagged_corr_for_weighting(tmp_path):
    from alpha_os.evaluation_runtime import apply_evaluation
    from alpha_os.meta_aggregation_service import refresh_target_meta_predictions
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("hyp_a")
        store.register_signal("hyp_b")

        history = [
            ("2026-03-24", 0.2, -0.2, 0.2),
            ("2026-03-25", 0.3, -0.3, 0.3),
            ("2026-03-26", 0.1, -0.1, 0.1),
        ]
        for date, pred_a, pred_b, obs in history:
            evaluation_id = f"BTC:residual_return_3d:{date}"
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="hyp_a",
                prediction_value=pred_a,
                observation_value=obs,
            )
            apply_evaluation(
                store,
                evaluation_id=evaluation_id,
                signal_id="hyp_b",
                prediction_value=pred_b,
                observation_value=obs,
            )

        current_evaluation_id = "BTC:residual_return_3d:2026-03-27"
        store.record_prediction(
            evaluation_id=current_evaluation_id,
            signal_id="hyp_a",
            prediction_value=1.0,
        )
        store.record_prediction(
            evaluation_id=current_evaluation_id,
            signal_id="hyp_b",
            prediction_value=0.0,
        )

        refresh_target_meta_predictions(store)
        items = {
            item.aggregation_kind: item
            for item in store.list_meta_predictions(limit=10)
            if item.evaluation_id == current_evaluation_id
        }
        assert items["active_equal_mean"].value == 0.5
        assert items["corr_weighted_mean"].value == 1.0
        assert items["corr_weighted_mean"].details is not None
        contributors = items["corr_weighted_mean"].details["contributors"]
        weights = {item["signal_id"]: item["weight"] for item in contributors}
        assert weights["hyp_a"] == 1.0
        assert weights["hyp_b"] == 0.0
    finally:
        store.close()


def test_compare_meta_aggregations_cli_orders_by_corr(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    for signal_id in ("momentum_1d", "reversal_1d", "average_gap_3d"):
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
        capsys.readouterr()

    values = [
        ("2026-03-24", "momentum_1d", "0.5", "0.2"),
        ("2026-03-24", "reversal_1d", "-0.5", "0.2"),
        ("2026-03-24", "average_gap_3d", "0.1", "0.2"),
        ("2026-03-25", "momentum_1d", "0.4", "0.1"),
        ("2026-03-25", "reversal_1d", "-0.4", "0.1"),
        ("2026-03-25", "average_gap_3d", "0.0", "0.1"),
        ("2026-03-26", "momentum_1d", "0.3", "0.05"),
        ("2026-03-26", "reversal_1d", "-0.3", "0.05"),
        ("2026-03-26", "average_gap_3d", "0.05", "0.05"),
    ]
    for date, signal_id, prediction, observation in values:
        assert (
            main(
                [
                    "debug-apply-evaluation",
                    "--db",
                    str(db_path),
                    "--date",
                    date,
                    "--signal-candidate-id",
                    signal_id,
                    "--prediction",
                    prediction,
                    "--observation",
                    observation,
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(
            [
                "debug-compare-meta-aggregations",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "alpha-os meta aggregation comparison" in output
    assert "residual_return_3d" in output
    assert "1. kind=" in output


def test_register_and_show_signal_specs_cli(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    assert (
        main(
            [
                "debug-register-signal-candidate-spec",
                "--db",
                str(db_path),
                "--signal-candidate-id",
                "reversal_1d_fast",
                "--base-signal-candidate-id",
                "reversal_1d",
            ]
        )
        == 0
    )
    register_output = capsys.readouterr().out
    assert "Signal Spec [created] reversal_1d_fast" in register_output
    assert "kind=reversal" in register_output
    assert "observable=daily_close" in register_output

    assert (
        main(
            [
                "debug-show-signal-candidate-specs",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    show_output = capsys.readouterr().out
    assert "alpha-os signal specs" in show_output
    assert "reversal_1d_fast" in show_output
    assert "observable=daily_close" in show_output


def test_register_and_show_observables_cli(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    assert (
        main(
            [
                "debug-show-observables",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    initial_output = capsys.readouterr().out
    assert "alpha-os observables" in initial_output
    assert "daily_close family=price value_kind=real_value resolution=1d" in initial_output

    assert (
        main(
            [
                "debug-register-observable",
                "--db",
                str(db_path),
                "--observable-id",
                "hlc3",
                "--family",
                "price",
                "--value-kind",
                "real_value",
                "--resolution",
                "1d",
            ]
        )
        == 0
    )
    register_output = capsys.readouterr().out
    assert "Observable [created] hlc3" in register_output

    assert (
        main(
            [
                "debug-show-observables",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    show_output = capsys.readouterr().out
    assert "hlc3 family=price value_kind=real_value resolution=1d" in show_output


def test_apply_and_inspect_runtime_manifest_cli(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "hlc3",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                        "params": {},
                    }
                ],
                "signal_specs": [
                    {
                        "signal_id": "reversal_hlc3",
                        "kind": "reversal",
                        "required_observable_id": "hlc3",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 1},
                    }
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_manifest",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_hlc3",
                                "observable_id": "hlc3",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_hlc3",
                            }
                        ],
                    }
                ],
                "signal_discoveries": [
                    {
                        "signal_discovery_id": "core_manifest_search",
                        "subject_set_id": "core_manifest",
                        "families": [
                            {
                                "kind": "reversal",
                                "parameter_space": {
                                    "lookback": [1],
                                },
                                "required_observable_id": "hlc3",
                                "target_id": "residual_return_3d",
                            }
                        ],
                        "target_id": "residual_return_3d",
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "core_manifest_eval",
                        "execution_range": {
                            "label": "manifest_exec",
                            "start_date": "2026-03-20",
                            "end_date": "2026-03-24",
                        },
                        "metric_group_names": [
                            "signal_discovery_run_stats",
                            "signal_discovery_result_stats",
                        ],
                    }
                ],
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:core_manifest_rule",
                            "label": "Core Manifest Rule",
                            "scope": {
                                "subject_set_id": "core_manifest",
                                "target_id": "residual_return_3d",
                            },
                            "signal_discovery_id": "core_manifest_search",
                            "position_rule_id": "signal_discovery",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="equal_weight",
                                direction_mode=None,
                                gross_exposure_cap=None,
                                rebalance_friction_policy={
                                    "turnover_friction": None,
                                    "no_trade_band": None,
                                },
                                execution_policy={
                                    "market_impact_bps": None,
                                    "fee_bps": None,
                                    "bid_ask_spread_bps": None,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "core_manifest_eval",
                        "strategy_id": "strategy:core_manifest_rule",
                        "base_url": "http://127.0.0.1:8000",
                        "portfolio_construction": {
                            "sizing_policy": {"sizing_method": "signal_weighted", "sizing_engine": "rule_based"}
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    apply_output = capsys.readouterr().out
    assert "Applied runtime manifest" in apply_output
    assert "InstrumentTypes: none" in apply_output
    assert "Observables:    total=1 created=1" in apply_output
    assert "Specifications: total=1 created=1" in apply_output
    assert "SubjectSets:    total=1 upserted=1" in apply_output
    assert "SignalDiscoveries: total=1 upserted=1" in apply_output
    assert "EvalSpecs:      total=1 upserted=1" in apply_output

    assert (
        main(
            [
                "inspect-runtime-resources",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    inspect_output = capsys.readouterr().out
    assert "alpha-os runtime resources" in inspect_output
    assert "hlc3 family=price value_kind=real_value resolution=1d" in inspect_output
    assert "reversal_hlc3" in inspect_output
    assert "core_manifest" in inspect_output
    assert "core_manifest_search" in inspect_output
    assert "core_manifest_eval" in inspect_output


def test_apply_runtime_manifest_supports_generated_discoveries(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "subject_sets": [
                    {
                        "subject_set_id": "us_equity_core",
                        "observation_specs": [
                            {
                                "observation_spec_id": "aaa_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "AAA_equity",
                                "subject_kind": "equity",
                                "asset": "AAA",
                                "observation_spec_id": "aaa_close",
                            }
                        ],
                    }
                ],
                "generated_signal_discoveries": [
                    {
                        "signal_discovery_id": "generated_core_search",
                        "subject_set_id": "us_equity_core",
                        "target_id": "residual_return_3d",
                        "operator_ids": [
                            "trend",
                            "volatility_breakout",
                            "relative_strength",
                        ],
                        "primary_observable_ids": [
                            "daily_close",
                            "cross_sectional_return_rank_20d",
                        ],
                        "conditioning_observable_ids": ["realized_vol_20d"],
                        "parameter_space": {
                            "lookback": [20, 40],
                        },
                        "constraint": {
                            "max_families_per_operator": 1,
                        },
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Applied runtime manifest" in output
    assert "generated_core_search" in output

    assert (
        main(
            [
                "inspect-runtime-resources",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    inspect_output = capsys.readouterr().out
    assert "generated_core_search" in inspect_output
    assert "cross_sectional_return_rank_20d" in inspect_output


def test_checked_in_narrow_manifest_applies_cleanly(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "runtime_manifests"
        / "us_equity_narrow_directional_context.json"
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "InstrumentTypes: none" in output
    assert "us_equity_narrow_core" in output
    assert "us_equity_narrow_directional_context_search" in output
    assert "us_equity_narrow_directional_context_eval" in output

    assert main(["inspect-runtime-resources", "--db", str(db_path)]) == 0
    inspect_output = capsys.readouterr().out
    assert "AAPL_equity" in inspect_output
    assert "ABNB_equity" in inspect_output
    assert "relative_strength" in inspect_output
    assert "us_equity_narrow_directional_context_search" in inspect_output


def test_checked_in_global_macro_manifest_applies_cleanly(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "runtime_manifests"
        / "global_macro_futures_daily_trend.json"
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "InstrumentTypes: future, perp" in output
    assert "global_macro_futures_26" in output
    assert "global_macro_futures_daily_trend_search" in output
    assert "global_macro_futures_daily_trend_eval" in output

    assert main(["inspect-runtime-resources", "--db", str(db_path)]) == 0
    inspect_output = capsys.readouterr().out
    assert "summary=[bindings=26 instruments=26" in inspect_output
    assert "instrument_types=future,perp" in inspect_output
    assert "ES_future" in inspect_output
    assert "BTCUSDT_perp" in inspect_output
    assert "time_series_trend" in inspect_output
    assert "global_macro_futures_daily_trend_search" in inspect_output

    store = EvaluationStore(db_path)
    try:
        case_states = store.list_evaluation_tasks(
            evaluation_spec_id="global_macro_futures_daily_trend_eval",
            limit=5,
        )
        assert len(case_states) == 1
        strategy_state = store.get_trading_strategy(case_states[0].task.strategy_id)
        assert strategy_state is not None
        construction = strategy_state.trading_strategy.portfolio_construction
        assert construction is not None
        assert construction.asset_class_weight_caps["equity_index"] == 0.55
        assert construction.cluster_weight_caps["eq_us"] == 0.2
    finally:
        store.close()


def test_checked_in_global_macro_tradeable_daily_10y_manifest_applies_cleanly(
    tmp_path, capsys
):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "runtime_manifests"
        / "global_macro_tradeable_daily_10y.json"
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "InstrumentTypes: future, perp" in output
    assert "global_macro_tradeable_daily_10y" in output
    assert "global_macro_tradeable_daily_10y_search" in output
    assert "global_macro_tradeable_daily_10y_eval" in output

    assert main(["inspect-runtime-resources", "--db", str(db_path)]) == 0
    inspect_output = capsys.readouterr().out
    assert "summary=[bindings=23 instruments=23" in inspect_output
    assert "instrument_types=future,perp" in inspect_output
    assert "YM_future" in inspect_output
    assert "HO_future" in inspect_output
    assert "BTCUSDT_perp" in inspect_output
    assert "SI_future" not in inspect_output
    assert "time_series_trend" in inspect_output
    assert "global_macro_tradeable_daily_10y_search" in inspect_output

    store = EvaluationStore(db_path)
    try:
        case_states = store.list_evaluation_tasks(
            evaluation_spec_id="global_macro_tradeable_daily_10y_eval",
            limit=5,
        )
        assert len(case_states) == 1
        strategy_state = store.get_trading_strategy(case_states[0].task.strategy_id)
        assert strategy_state is not None
        construction = strategy_state.trading_strategy.portfolio_construction
        assert construction is not None
        assert construction.sizing_method == "diversified_risk_budget"
        assert construction.sizing_engine == "history_based"
        assert construction.long_only is False
        assert construction.active_overlay is not None
        assert construction.active_overlay.active_weight_budget == 0.0
        assert construction.target_vol == 0.18
        assert construction.asset_class_weight_caps["crypto"] == 0.2
        assert construction.portfolio_intent.effective_n_floor == 8.0
        assert construction.portfolio_intent.top_gross_share_cap_n == 3
        assert construction.portfolio_intent.top_gross_share_cap == 0.55
        assert construction.sleeve_composition is not None
        assert [
            sleeve.sleeve_id
            for sleeve in construction.sleeve_composition.sleeves
            if sleeve.enabled
        ] == ["trend_core"]
    finally:
        store.close()


def test_checked_in_us_etf_dual_momentum_10y_manifest_applies_cleanly(
    tmp_path, capsys
):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "runtime_manifests"
        / "us_etf_broad_dual_momentum_10y.json"
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "us_etf_broad_multi_asset" in output
    assert "us_etf_broad_dual_momentum_search" in output
    assert "us_etf_broad_dual_momentum_eval_10y" in output

    assert main(["inspect-runtime-resources", "--db", str(db_path)]) == 0
    inspect_output = capsys.readouterr().out
    assert "base_currency=USD" in inspect_output
    assert "trading_calendar=XNYS" in inspect_output
    assert "benchmark_id=us_etf_broad_multi_asset" in inspect_output
    assert "trend" in inspect_output
    assert "relative_strength" in inspect_output
    assert "folds=10" in inspect_output

    store = EvaluationStore(db_path)
    try:
        case_states = store.list_evaluation_tasks(
            evaluation_spec_id="us_etf_broad_dual_momentum_eval_10y",
            limit=5,
        )
        assert len(case_states) == 1
        strategy_state = store.get_trading_strategy(case_states[0].task.strategy_id)
        assert strategy_state is not None
        strategy = strategy_state.trading_strategy
        construction = strategy.portfolio_construction
        assert construction is not None
        assert construction.rebalance_interval_steps == 21
        assert strategy.portfolio.top_k == 3
        assert construction.target_vol == 0.1
        assert construction.gross_leverage_cap == 1.0
        assert construction.net_exposure_target == 1.0
    finally:
        store.close()


def test_apply_runtime_manifest_rejects_incomplete_multi_subject_universe_policy(
    tmp_path,
    capsys,
):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "subject_sets": [
                    {
                        "subject_set_id": "macro_pair",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            },
                            {
                                "observation_spec_id": "eth_close",
                                "observable_id": "daily_close",
                            },
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            },
                            {
                                "subject_id": "ETH_spot",
                                "asset": "ETH",
                                "observation_spec_id": "eth_close",
                            },
                        ],
                        "universe_policy": {
                            "base_currency": "USD",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
    assert exc_info.value.code == 2
    assert "subject set universe policy is incomplete" in capsys.readouterr().err


def test_rebuild_strategy_adaptation_state_from_evaluation_report(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.evaluation_report import (
        EvaluationTaskResult,
        EvaluationMetricGroupResult,
        EvaluationReport,
    )
    from alpha_os.screening import (
        ScreeningCandidateResult,
        ScreeningPolicy,
        ScreeningResult,
    )
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        screening_result = ScreeningResult(
            screening_result_id="discovery:screen",
            signal_discovery_id="discovery",
            policy=ScreeningPolicy(),
            candidates=(
                ScreeningCandidateResult(
                    signal_id="trend@AAPL_equity",
                    specification_signal_id="trend",
                    family_id="trend_family",
                    subject_id="AAPL_equity",
                    target_id="residual_return_3d",
                    kind="momentum",
                    lookback=40,
                    score=0.2,
                    corr=0.2,
                    stability_score=0.9,
                    sample_count=20,
                    keep=True,
                    family_rank=1,
                    reasons=(),
                ),
            ),
            created_at="2026-04-03T00:00:00+00:00",
        )
        store.upsert_screening_result(result=screening_result)
        report = EvaluationReport(
            evaluation_report_id="report-1",
            evaluation_spec_id="protocol-1",
            task_results=(
                EvaluationTaskResult(
                    evaluation_task_id="case:discovery",
                    strategy_id="strategy:discovery",
                    signal_discovery_id="discovery",
                    metric_group_results=(
                        EvaluationMetricGroupResult(
                            metric_group_name="signed_belief_quality",
                            source="native_plan",
                            metrics={"mean_survivor_corr": 0.2},
                        ),
                    ),
                    failure_finding_groups=(),
                    artifact_refs={"screening_result_ids": ("discovery:screen",)},
                ),
            ),
            created_at="2026-04-03T00:00:00+00:00",
        )
        store.upsert_evaluation_report(report=report)
    finally:
        store.close()

    assert (
        main(
            [
                "rebuild-strategy-adaptation-state",
                "--db",
                str(db_path),
                "--report-id",
                "report-1",
                "--strategy-id",
                "strategy:discovery",
            ]
        )
        == 0
    )
    refresh_output = capsys.readouterr().out
    assert "alpha-os strategy adaptation states" in refresh_output
    assert "trend_family" in refresh_output

    assert (
        main(
            [
                "show-strategy-adaptation-states",
                "--db",
                str(db_path),
                "--strategy-id",
                "strategy:discovery",
            ]
        )
        == 0
    )
    show_output = capsys.readouterr().out
    assert "report-1" in show_output
    assert "family=trend_family" in show_output


def test_compress_screening_result_only_applies_strategy_adaptation_state_when_strategy_is_explicit():
    from types import SimpleNamespace

    from alpha_os.signal_discovery_application import compress_screening_result_state
    from alpha_os.strategy_adaptation import (
        FamilyReputation,
        StrategyAdaptationState,
        SignalReputation,
    )
    from alpha_os.trading_strategy import AdaptationPolicySpec
    from alpha_os.screening import (
        ScreeningCandidateResult,
        ScreeningPolicy,
        ScreeningResult,
    )

    screening_result = ScreeningResult(
        screening_result_id="discovery:screen",
        signal_discovery_id="discovery",
        policy=ScreeningPolicy(),
        candidates=(
            ScreeningCandidateResult(
                signal_id="trend_fast@AAPL_equity",
                specification_signal_id="trend_fast",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                kind="trend",
                lookback=20,
                score=0.5,
                corr=0.5,
                stability_score=1.5,
                sample_count=20,
                keep=True,
                family_rank=1,
                reasons=(),
            ),
            ScreeningCandidateResult(
                signal_id="trend_slow@AAPL_equity",
                specification_signal_id="trend_slow",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                kind="trend",
                lookback=60,
                score=0.4,
                corr=0.4,
                stability_score=1.5,
                sample_count=20,
                keep=True,
                family_rank=2,
                reasons=(),
            ),
        ),
        created_at="2026-04-03T00:00:00+00:00",
    )
    strategy_adaptation_state = StrategyAdaptationState(
        strategy_id="strategy:test",
        signal_discovery_id="discovery",
        source_evaluation_report_id="report-1",
        source_screening_result_id="discovery:screen",
        signal_reputations=(
            SignalReputation(
                signal_id="trend_fast@AAPL_equity",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                orientation=1,
                edge_score=0.60,
                mmc=0.12,
                contribution_score=0.36,
                confidence=0.90,
                stability_score=1.5,
                sample_count=20,
                update_count=2,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
            SignalReputation(
                signal_id="trend_slow@AAPL_equity",
                family_id="trend_family",
                subject_id="AAPL_equity",
                target_id="residual_return_3d",
                orientation=1,
                edge_score=0.20,
                mmc=0.01,
                contribution_score=0.105,
                confidence=0.70,
                stability_score=1.2,
                sample_count=20,
                update_count=1,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
        ),
        family_reputations=(
            FamilyReputation(
                family_id="trend_family",
                mean_edge_score=0.40,
                mean_confidence=0.80,
                mean_stability_score=1.35,
                subject_coverage=1,
                member_count=2,
                update_count=2,
                updated_at="2026-04-03T00:00:00+00:00",
            ),
        ),
        created_at="2026-04-03T00:00:00+00:00",
    )

    class DummyStore:
        def __init__(self):
            self.persisted_beliefs = []

        def get_screening_result(self, screening_result_id):
            assert screening_result_id == "discovery:screen"
            return SimpleNamespace(result=screening_result)

        def list_latest_evaluation_snapshots(self, *, signal_ids):
            assert set(signal_ids) == {
                "trend_fast@AAPL_equity",
                "trend_slow@AAPL_equity",
            }
            return [
                SimpleNamespace(
                    signal_id="trend_fast@AAPL_equity",
                    prediction_value=0.10,
                ),
                SimpleNamespace(
                    signal_id="trend_slow@AAPL_equity",
                    prediction_value=0.02,
                ),
            ]

        def get_strategy_adaptation_state(self, strategy_id):
            if strategy_id == "strategy:test":
                return SimpleNamespace(state=strategy_adaptation_state)
            return None

        def get_trading_strategy(self, strategy_id):
            if strategy_id == "strategy:test":
                return SimpleNamespace(
                    trading_strategy=SimpleNamespace(
                        adaptation_policy=AdaptationPolicySpec(
                            enabled=True,
                            adaptation_blend=0.2,
                        )
                    )
                )
            if strategy_id == "strategy:disabled":
                return SimpleNamespace(
                    trading_strategy=SimpleNamespace(
                        adaptation_policy=AdaptationPolicySpec(
                            enabled=False,
                            adaptation_blend=0.2,
                        )
                    )
                )
            return None

        def upsert_compressed_belief(self, *, belief):
            self.persisted_beliefs.append(belief)
            return SimpleNamespace(belief=belief)

    store = DummyStore()

    baseline = compress_screening_result_state(
        store,
        screening_result_id="discovery:screen",
    )
    adapted = compress_screening_result_state(
        store,
        screening_result_id="discovery:screen",
        strategy_id="strategy:test",
    )
    disabled = compress_screening_result_state(
        store,
        screening_result_id="discovery:screen",
        strategy_id="strategy:disabled",
    )

    assert round(baseline.belief.components[0].belief_value, 6) == 0.064444
    assert round(adapted.belief.components[0].belief_value, 6) == 0.06506
    assert round(disabled.belief.components[0].belief_value, 6) == 0.064444


def test_subject_bound_executable_rejects_mismatched_observable():
    import pytest

    from alpha_os.signal_registry import (
        SignalSpec,
        build_subject_bound_signal_definition,
    )
    from alpha_os.portfolio_decision import ObservationSpec

    with pytest.raises(ValueError, match="required_observable_id=hlc3"):
        build_subject_bound_signal_definition(
            specification=SignalSpec(
                signal_id="reversal_hlc3",
                kind="reversal",
                lookback=1,
                required_observable_id="hlc3",
            ),
            subject_id="BTC_spot",
            asset="BTC",
            observation_spec=ObservationSpec(
                observation_spec_id="btc_close",
                observable_id="daily_close",
                ),
        )


def test_subject_bound_executable_accepts_derived_observable_from_daily_close():
    from alpha_os.signal_registry import (
        SignalSpec,
        build_subject_bound_signal_definition,
    )
    from alpha_os.portfolio_decision import ObservationSpec
    from alpha_os.targets import residual_return_target_definition

    definition = build_subject_bound_signal_definition(
        specification=SignalSpec(
            signal_id="reversal_daily_return",
            kind="reversal",
            lookback=20,
            required_observable_id="daily_return",
            target=residual_return_target_definition(3),
        ),
        subject_id="BTC_spot",
        asset="BTC",
        observation_spec=ObservationSpec(
            observation_spec_id="btc_close",
            observable_id="daily_close",
        ),
    )

    assert definition.signal_id == "reversal_daily_return@BTC_spot"
    assert definition.observation_spec is not None
    assert definition.observation_spec.observable_id == "daily_close"


def test_subject_bound_executable_accepts_cross_sectional_observable_from_daily_close():
    from alpha_os.signal_registry import (
        SignalSpec,
        build_subject_bound_signal_definition,
    )
    from alpha_os.portfolio_decision import ObservationSpec
    from alpha_os.targets import residual_return_target_definition

    definition = build_subject_bound_signal_definition(
        specification=SignalSpec(
            signal_id="relative_strength_rank_20d",
            kind="relative_strength_rank",
            lookback=20,
            required_observable_id="cross_sectional_return_rank_20d",
            target=residual_return_target_definition(3),
        ),
        subject_id="AAPL_equity",
        asset="AAPL",
        observation_spec=ObservationSpec(
            observation_spec_id="aapl_close",
            observable_id="daily_close",
        ),
    )

    assert definition.signal_id == "relative_strength_rank_20d@AAPL_equity"


def test_backfill_subject_set_can_pre_screen_before_full_materialization(tmp_path, capsys):
    import pandas as pd

    from alpha_os import data_repositories as data_repositories_module
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"

    def _fake_loader(*, base_url: str, asset: str | None = None, observation_spec=None, signal_name=None):
        assert base_url == "http://example.com"
        assert asset == "BTC"
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-20T00:00:00Z",
                    "2026-03-21T00:00:00Z",
                    "2026-03-22T00:00:00Z",
                    "2026-03-23T00:00:00Z",
                    "2026-03-24T00:00:00Z",
                    "2026-03-25T00:00:00Z",
                    "2026-03-26T00:00:00Z",
                    "2026-03-27T00:00:00Z",
                ],
                "value": [100.0, 104.0, 101.0, 106.0, 103.0, 109.0, 104.0, 111.0],
            }
        )

    for signal_id in ("reversal_1d", "reversal_3d", "average_gap_3d"):
        assert (
            main(
                [
                    "debug-register-signal-candidate-spec",
                    "--db",
                    str(db_path),
                    "--signal-candidate-id",
                    signal_id,
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto",
                "--observation-spec",
                "btc_close=daily_close",
                "--subject-binding",
                "BTC_spot=BTC=btc_close",
            ]
        )
        == 0
    )
    capsys.readouterr()

    original_loader = data_repositories_module.load_observation_frame
    data_repositories_module.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "debug-backfill-subject-set",
                    "--db",
                    str(db_path),
                    "--start-date",
                    "2026-03-23",
                    "--end-date",
                    "2026-03-24",
                    "--subject-set-id",
                    "core_crypto",
                    "--signal-candidate-spec-id",
                    "reversal_1d",
                    "--signal-candidate-spec-id",
                    "reversal_3d",
                    "--signal-candidate-spec-id",
                    "average_gap_3d",
                    "--pre-screen-top-k-per-kind",
                    "1",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories_module.load_observation_frame = original_loader

    output = capsys.readouterr().out
    assert "pre_screen_selected=2/3" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        snapshots = store.list_evaluation_snapshots(limit=20)
        assert len(snapshots) == 4
    finally:
        store.close()


def test_build_and_show_portfolio_decisions_cli(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"

    for signal_id in ("reversal_1d", "average_gap_3d"):
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
        capsys.readouterr()

    values = [
        ("2026-03-24", "reversal_1d", "0.4", "0.2"),
        ("2026-03-24", "average_gap_3d", "0.0", "0.2"),
        ("2026-03-25", "reversal_1d", "0.3", "0.1"),
        ("2026-03-25", "average_gap_3d", "0.1", "0.1"),
        ("2026-03-26", "reversal_1d", "0.2", "0.05"),
        ("2026-03-26", "average_gap_3d", "0.0", "0.05"),
    ]
    for date, signal_id, prediction, observation in values:
        assert (
            main(
                [
                    "debug-apply-evaluation",
                    "--db",
                    str(db_path),
                    "--date",
                    date,
                    "--signal-candidate-id",
                    signal_id,
                    "--prediction",
                    prediction,
                    "--observation",
                    observation,
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(
                [
                "debug-decide-portfolio-runtime",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--subject-id",
                "BTC_spot",
                "--capital-base",
                "2.5",
                "--gross-exposure-cap",
                "0.5",
                "--gross-limit",
                "0.6",
                "--net-limit",
                "0.2",
                "--rebalance-step",
                "3",
                "--market-impact-bps",
                "25",
                "--no-trade-band",
                "0.01",
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert "alpha-os portfolio decisions" in build_output
    assert "portfolio=paper_core" in build_output
    assert "subject=BTC_spot" in build_output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        decisions = store.list_portfolio_decisions(portfolio_id="paper_core", limit=10)
        assert decisions[0].details is not None
        assert decisions[0].details["portfolio_state"]["capital_base"] == 2.5
        assert decisions[0].details["portfolio_state"]["gross_limit"] == 0.6
        assert decisions[0].details["portfolio_state"]["net_limit"] == 0.2
        assert decisions[0].details["portfolio_state"]["rebalance_step"] == 3
    finally:
        store.close()

    assert (
        main(
            [
                "debug-show-portfolio-decisions",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--details",
            ]
        )
        == 0
    )
    show_output = capsys.readouterr().out
    assert "alpha-os portfolio decisions" in show_output
    assert "kind=corr_weighted_mean" in show_output
    assert "target=residual_return_3d" in show_output
    assert "sizing=signal_weighted engine=rule_based" in show_output
    assert "cost={" in show_output
    assert "uncertainty={" in show_output


def test_build_portfolio_decision_cli_supports_multi_subject_inputs(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_meta_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-26",
            asset="BTC",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.2,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.2},{"prediction":0.1}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="BTC",
            target_id="residual_return_3d",
            corr=0.3,
            sample_count=10,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        store.upsert_meta_prediction(
            evaluation_id="ETH:residual_return_3d:2026-03-26",
            asset="ETH",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.1,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.1},{"prediction":0.05}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="ETH",
            target_id="residual_return_3d",
            corr=0.2,
            sample_count=8,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        for date, btc_obs, eth_obs in (
            ("2026-03-24", 0.10, 0.20),
            ("2026-03-25", 0.20, 0.40),
            ("2026-03-26", 0.05, 0.10),
        ):
            store.finalize_observation(
                evaluation_id=f"BTC:residual_return_3d:{date}",
                observation_value=btc_obs,
                asset="BTC",
                target_id="residual_return_3d",
            )
            store.finalize_observation(
                evaluation_id=f"ETH:residual_return_3d:{date}",
                observation_value=eth_obs,
                asset="ETH",
                target_id="residual_return_3d",
            )
    finally:
        store.close()

    assert (
        main(
                [
                "debug-decide-portfolio-runtime",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--subject-id",
                "BTC_spot",
                "--observation-spec",
                "eth_close=daily_close",
                "--subject-binding",
                "ETH_spot=ETH=eth_close",
                "--sizing-method",
                "signal_weighted",
                "--sizing-engine",
                "optimizer",
                "--gross-exposure-cap",
                "0.5",
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert "alpha-os portfolio decisions" in build_output
    assert "subject=BTC_spot" in build_output
    assert "subject=ETH_spot" in build_output

    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto",
                "--observation-spec",
                "btc_close=daily_close",
                "--observation-spec",
                "eth_close=daily_close",
                "--subject-binding",
                "BTC_spot=BTC=btc_close",
                "--subject-binding",
                "ETH_spot=ETH=eth_close",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "debug-show-portfolio-decisions",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--subject-set-id",
                "core_crypto",
            ]
        )
        == 0
    )
    show_output = capsys.readouterr().out
    assert "SubjectSet: core_crypto" in show_output
    assert "subject=BTC_spot" in show_output
    assert "subject=ETH_spot" in show_output


def test_debug_decide_portfolio_runtime_uses_strategy_scope_and_constraints(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
        UniversePolicySpec,
    )
    from alpha_os.store import EvaluationStore
    from alpha_os.trading_strategy import (
        AdaptationPolicySpec,
        ExecutionPolicySpec,
        HoldingCostPolicySpec,
        RebalanceFrictionPolicySpec,
        StrategyPortfolioSpec,
        TradingStrategyScopeSpec,
        TradingStrategySpec,
    )
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_meta_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-26",
            asset="BTC",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.2,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.2},{"prediction":0.1}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="BTC",
            target_id="residual_return_3d",
            corr=0.3,
            sample_count=10,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        store.upsert_meta_prediction(
            evaluation_id="ETH:residual_return_3d:2026-03-26",
            asset="ETH",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.1,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.1},{"prediction":0.05}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="ETH",
            target_id="residual_return_3d",
            corr=0.2,
            sample_count=8,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        for date, btc_obs, eth_obs in (
            ("2026-03-24", 0.10, 0.20),
            ("2026-03-25", 0.20, 0.40),
            ("2026-03-26", 0.05, 0.10),
        ):
            store.finalize_observation(
                evaluation_id=f"BTC:residual_return_3d:{date}",
                observation_value=btc_obs,
                asset="BTC",
                target_id="residual_return_3d",
            )
            store.finalize_observation(
                evaluation_id=f"ETH:residual_return_3d:{date}",
                observation_value=eth_obs,
                asset="ETH",
                target_id="residual_return_3d",
            )
        store.upsert_subject_set(
            "core_crypto",
            definition=SubjectSet(
                subject_set_id="core_crypto",
                observation_specs=(
                    ObservationSpec(
                        observation_spec_id="btc_close",
                        observable_id="daily_close",
                    ),
                    ObservationSpec(
                        observation_spec_id="eth_close",
                        observable_id="daily_close",
                    ),
                ),
                bindings=(
                    SubjectObservationBinding(
                        subject_id="BTC_spot",
                        asset="BTC",
                        observation_spec_id="btc_close",
                    ),
                    SubjectObservationBinding(
                        subject_id="ETH_spot",
                        asset="ETH",
                        observation_spec_id="eth_close",
                    ),
                ),
                universe_policy=UniversePolicySpec(
                    base_currency="USD",
                    trading_calendar="multi_venue",
                    benchmark_id="core_crypto",
                ),
            )
        )
        store.upsert_trading_strategy(
            trading_strategy=TradingStrategySpec(
                strategy_id="strategy:paper_core_top1",
                label="Paper Core Top 1",
                scope=TradingStrategyScopeSpec(
                    subject_set_id="core_crypto",
                    target_id="residual_return_3d",
                ),
                signal_discovery_id=None,
                position_rule_id="constant_hold",
                family_mix=None,
                portfolio=StrategyPortfolioSpec(
                    portfolio_construction=PortfolioConstructionSpec(
                        sizing_policy=PortfolioConstructionSizingSpec(
                            sizing_method="equal_weight",
                        ),
                        direction_mode="long_only",
                        gross_exposure_cap=0.5,
                    ),
                    rebalance_friction_policy=RebalanceFrictionPolicySpec(
                        turnover_friction=0.15,
                        no_trade_band=0.02,
                    ),
                    execution_policy=ExecutionPolicySpec(
                        market_impact_bps=7.0,
                        fee_bps=1.5,
                    ),
                    holding_cost_policy=HoldingCostPolicySpec(
                        funding_bps_per_step=2.5,
                    ),
                    selection_kind="top_k",
                    top_k=1,
                ),
                adaptation_policy=AdaptationPolicySpec(
                    enabled=False,
                    adaptation_blend=0.2,
                ),
                created_at="2026-04-17T00:00:00Z",
            )
        )
    finally:
        store.close()

    assert (
        main(
            [
                "debug-decide-portfolio-runtime",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--strategy-id",
                "strategy:paper_core_top1",
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert "Strategy: strategy:paper_core_top1" in build_output
    assert "selection=top_k" in build_output
    assert "sizing=equal_weight engine=history_based" in build_output
    assert "rebalance=every_1_steps" in build_output
    assert "subject=BTC_spot" in build_output
    assert "subject=ETH_spot" in build_output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        decisions = store.list_portfolio_decisions(portfolio_id="paper_core", limit=10)
        assert {item.subject_id for item in decisions} == {"BTC_spot", "ETH_spot"}
        weights = {item.subject_id: item.target_weight for item in decisions}
        assert all(value >= 0.0 for value in weights.values())
        assert sum(1 for value in weights.values() if value > 0.0) == 1
        assert sum(abs(value) for value in weights.values()) <= 0.5000001
        details = decisions[0].details
        assert details is not None
        assert details["sizing_method"] == "equal_weight"
        assert details["sizing_engine"] == "history_based"
        strategy_details = details["strategy"]
        assert strategy_details is not None
        assert strategy_details["strategy_id"] == "strategy:paper_core_top1"
        assert strategy_details["selection_kind"] == "top_k"
        assert strategy_details["rebalance"] == "every_1_steps"
        assert strategy_details["top_k"] == 1
        cost_inputs = details["assumptions"]["cost_inputs"]
        names = {item["name"] for item in cost_inputs}
        assert {"turnover_friction", "market_impact", "fee_bps", "funding_bps_per_step", "no_trade_band"} <= names
        turnover_item = next(item for item in cost_inputs if item["name"] == "turnover_friction")
        funding_item = next(item for item in cost_inputs if item["name"] == "funding_bps_per_step")
        assert turnover_item["value"] == 0.15
        assert funding_item["value"] == 2.5
    finally:
        store.close()

    assert (
        main(
            [
                "debug-show-portfolio-decisions",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--details",
            ]
        )
        == 0
    )
    show_output = capsys.readouterr().out
    assert "strategy=strategy:paper_core_top1" in show_output
    assert "selection=top_k" in show_output
    assert "rebalance=every_1_steps" in show_output
    assert "sizing=equal_weight engine=history_based" in show_output


def test_multi_subject_runtime_e2e_from_subject_set_backfill(tmp_path, capsys):
    import alpha_os.data_repositories as data_repositories
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        frames = {
            "BTC": pd.DataFrame(
                {
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
            ),
            "ETH": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                    ],
                    "value": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0],
                }
            ),
        }
        return frames[asset]

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "debug-register-signal-candidate-spec",
                    "--db",
                    str(db_path),
                    "--signal-candidate-id",
                    "reversal_1d",
                ]
            )
            == 0
        )
        capsys.readouterr()
        assert (
            main(
                [
                    "debug-register-subject-set",
                    "--db",
                    str(db_path),
                    "--subject-set-id",
                    "core_crypto",
                    "--observation-spec",
                    "btc_close=daily_close",
                    "--observation-spec",
                    "eth_close=daily_close",
                    "--subject-binding",
                    "BTC_spot=BTC=btc_close",
                    "--subject-binding",
                    "ETH_spot=ETH=eth_close",
                ]
            )
            == 0
        )
        capsys.readouterr()
        assert (
            main(
                [
                    "debug-backfill-subject-set",
                    "--db",
                    str(db_path),
                    "--start-date",
                    "2026-03-22",
                    "--end-date",
                    "2026-03-22",
                    "--subject-set-id",
                    "core_crypto",
                    "--signal-candidate-spec-id",
                    "reversal_1d",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
        capsys.readouterr()
    finally:
        data_repositories.load_observation_frame = original_loader

    assert (
        main(
                [
                "debug-decide-portfolio-runtime",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--subject-id",
                "BTC_spot",
                "--subject-set-id",
                "core_crypto",
                "--sizing-method",
                "signal_weighted",
                "--sizing-engine",
                "optimizer",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "subject=BTC_spot" in output
    assert "subject=ETH_spot" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        btc_metrics = store.list_meta_prediction_metrics(asset="BTC", target_id="residual_return_3d")
        eth_metrics = store.list_meta_prediction_metrics(asset="ETH", target_id="residual_return_3d")
        decisions = store.list_portfolio_decisions(portfolio_id="paper_core", limit=10)
        assert btc_metrics
        assert eth_metrics
        assert {item.subject_id for item in decisions} == {"BTC_spot", "ETH_spot"}
    finally:
        store.close()


def test_apply_subject_set_backfill_registers_executables_and_builds_runtime(tmp_path, capsys):
    import alpha_os.data_repositories as data_repositories
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        frames = {
            "BTC": pd.DataFrame(
                {
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
            ),
            "ETH": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                    ],
                    "value": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0],
                }
            ),
        }
        return frames[asset]

    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto",
                "--observation-spec",
                "btc_close=daily_close",
                "--observation-spec",
                "eth_close=daily_close",
                "--subject-binding",
                "BTC_spot=BTC=btc_close",
                "--subject-binding",
                "ETH_spot=ETH=eth_close",
            ]
        )
        == 0
    )
    capsys.readouterr()

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "debug-backfill-subject-set",
                    "--db",
                    str(db_path),
                    "--start-date",
                    "2026-03-22",
                    "--end-date",
                    "2026-03-22",
                    "--subject-set-id",
                    "core_crypto",
                    "--signal-candidate-spec-id",
                    "reversal_1d",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader
    output = capsys.readouterr().out
    assert "subject_set=core_crypto" in output
    assert "reversal_1d@BTC_spot" in output
    assert "reversal_1d@ETH_spot" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        signals = (
            store.list_signals(asset="BTC")
            + store.list_signals(asset="ETH")
        )
        assert {item.signal_id for item in signals} == {
            "reversal_1d@BTC_spot",
            "reversal_1d@ETH_spot",
        }
        btc_metrics = store.list_meta_prediction_metrics(asset="BTC", target_id="residual_return_3d")
        eth_metrics = store.list_meta_prediction_metrics(asset="ETH", target_id="residual_return_3d")
        assert btc_metrics
        assert eth_metrics
    finally:
        store.close()


def test_screen_discovery_persists_survivors(tmp_path, capsys):
    import alpha_os.data_repositories as data_repositories
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "observables": [
                    {
                        "observable_id": "daily_close",
                        "family": "price",
                        "value_kind": "real_value",
                        "default_resolution": "1d",
                    }
                ],
                "signal_specs": [
                    {
                        "signal_id": "reversal_1d",
                        "kind": "reversal",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 1},
                    },
                    {
                        "signal_id": "reversal_3d",
                        "kind": "reversal",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 3},
                    },
                    {
                        "signal_id": "average_gap_3d",
                        "kind": "average_gap",
                        "required_observable_id": "daily_close",
                        "target_definition": {
                            "target_id": "residual_return_3d",
                            "family": "residual_return",
                            "observation_kind": "realized_return",
                            "subject_kind": "asset",
                            "output_kind": "real_value",
                            "scoring_kind": "corr",
                            "horizon_days": 3,
                            "params": {},
                        },
                        "params": {"lookback": 3},
                    },
                ],
                "subject_sets": [
                    {
                        "subject_set_id": "core_crypto",
                        "observation_specs": [
                            {
                                "observation_spec_id": "btc_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "BTC_spot",
                                "subject_kind": "asset",
                                "asset": "BTC",
                                "observation_spec_id": "btc_close",
                            }
                        ],
                    }
                ],
                "signal_discoveries": [
                    {
                        "signal_discovery_id": "core_crypto_search",
                        "subject_set_id": "core_crypto",
                        "selection_policy": {
                            "min_sample_count": 1,
                            "min_abs_corr": 0.0,
                            "probe_max_dates": 3,
                            "probe_min_sample_count": 2,
                            "probe_min_abs_corr": 0.0,
                            "probe_max_family_survivors_per_subject": 1,
                            "survivor_min_sample_count": 2,
                            "survivor_min_abs_corr": 0.0,
                            "survivor_max_family_survivors_per_subject": 1,
                            "snapshot_retention": "latest_per_survivor",
                            "max_family_survivors_per_subject": 2,
                        },
                        "families": [
                            {
                                "family_id": "reversal_family",
                                "kind": "reversal",
                                "parameter_space": {
                                    "lookback": [1, 3],
                                },
                                "required_observable_id": "daily_close",
                                "target_id": "residual_return_3d",
                                "survivor_budget": 1,
                            },
                            {
                                "family_id": "average_gap_family",
                                "kind": "average_gap",
                                "parameter_space": {
                                    "lookback": [3],
                                },
                                "required_observable_id": "daily_close",
                                "target_id": "residual_return_3d",
                            },
                        ],
                        "target_id": "residual_return_3d",
                    }
                ],
                "evaluation_specs": [
                    {
                        "evaluation_spec_id": "core_crypto_eval",
                        "execution_range": {
                            "label": "exec_window",
                            "start_date": "2026-03-23",
                            "end_date": "2026-03-24",
                        },
                        "evaluation_date_ranges": [
                            {
                                "label": "eval_window",
                                "start_date": "2026-03-23",
                                "end_date": "2026-03-24",
                            }
                        ],
                        "metric_windows": [2],
                        "metric_group_names": [
                            "signal_discovery_run_stats",
                            "signal_discovery_result_stats",
                            "signed_belief_quality",
                            "portfolio_target_return_alignment",
                            "sizing_policy_quality",
                            "rebalance_policy_quality",
                            "decision_quality",
                            "robustness",
                        ],
                    }
                ],
                "strategy_specs": [
                    {
                        "trading_strategy": {
                            "strategy_id": "strategy:core_crypto_rule",
                            "label": "Core Crypto Rule",
                            "scope": {
                                "subject_set_id": "core_crypto",
                                "target_id": "residual_return_3d",
                            },
                            "signal_discovery_id": "core_crypto_search",
                            "position_rule_id": "signal_discovery",
                            "family_mix": None,
                            **_strategy_portfolio_document(
                                sizing_method="signal_weighted",
                                direction_mode=None,
                                gross_exposure_cap=None,
                                rebalance_friction_policy={
                                    "turnover_friction": None,
                                    "no_trade_band": None,
                                },
                                execution_policy={
                                    "market_impact_bps": None,
                                    "fee_bps": None,
                                    "bid_ask_spread_bps": None,
                                },
                            ),
                            "created_at": "2026-04-05T00:00:00+00:00",
                        }
                    }
                ],
                "evaluation_tasks": [
                    {
                        "evaluation_spec_id": "core_crypto_eval",
                        "strategy_id": "strategy:core_crypto_rule",
                        "base_url": "http://127.0.0.1:8000",
                        "portfolio_construction": {
                            "sizing_policy": {"sizing_method": "signal_weighted", "sizing_engine": "rule_based"}
                        },
                        "rebalance_friction_policy": {},
                        "execution_cost_assumptions": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "apply-manifest",
                "--db",
                str(db_path),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        assert asset == "BTC"
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-20T00:00:00Z",
                    "2026-03-21T00:00:00Z",
                    "2026-03-22T00:00:00Z",
                    "2026-03-23T00:00:00Z",
                    "2026-03-24T00:00:00Z",
                    "2026-03-25T00:00:00Z",
                    "2026-03-26T00:00:00Z",
                    "2026-03-27T00:00:00Z",
                ],
                "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 107.0, 106.0],
            }
        )

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                    [
                        "debug-run-signal-discovery",
                        "--db",
                        str(db_path),
                        "--start-date",
                        "2026-03-23",
                        "--end-date",
                        "2026-03-24",
                        "--signal-discovery-id",
                        "core_crypto_search",
                        "--base-url",
                        "http://example.com",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader
    output = capsys.readouterr().out
    assert "signal_discovery=core_crypto_search" in output
    assert "executables=2" in output
    assert "pre_screen_selected=3/3" in output
    assert "probe_selected=2/3" in output
    assert "survivor_selected=2/2" in output
    assert "alpha-os screening" in output
    assert "SignalDiscovery: core_crypto_search" in output
    assert "Candidates:  total=2 survivors=2" in output
    assert "family=reversal_family" in output
    assert "family=average_gap_family" in output
    assert "alpha-os compressed belief" in output
    assert "subject=BTC_spot" in output
    assert "cluster_count=" in output
    assert "effective_beliefs=" in output
    assert "diversity=" in output
    assert "representatives=" in output
    assert "alpha-os workflow output" in output
    assert "CompressedBeliefId:" in output
    assert "PrunedSnapshots:" in output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        signals = store.list_signals(subject_id="BTC_spot", target_id="residual_return_3d")
        signal_ids = {item.signal_id for item in signals}
        assert "average_gap_3d@BTC_spot" in signal_ids
        assert len([item for item in signal_ids if item.startswith("reversal_")]) == 1
        assert len(signal_ids) == 2
        screening_results = store.list_screening_results(
            signal_discovery_id="core_crypto_search",
        )
        assert len(screening_results) == 1
        assert len(screening_results[0].result.survivors) == 2
        survivor_family_ids = {
            item.family_id for item in screening_results[0].result.survivors
        }
        assert survivor_family_ids == {"reversal_family", "average_gap_family"}
        beliefs = store.list_compressed_beliefs(signal_discovery_id="core_crypto_search")
        assert len(beliefs) == 1
        assert len(beliefs[0].belief.components) == 1
        assert beliefs[0].belief.components[0].subject_id == "BTC_spot"
        assert beliefs[0].belief.components[0].family_count == 2
        assert beliefs[0].belief.components[0].cluster_count >= 1
        assert beliefs[0].belief.components[0].effective_belief_count >= 1.0
        assert beliefs[0].belief.components[0].diversity_score > 0.0
        assert len(store.list_evaluation_snapshots(limit=10)) == 2
        compressed_belief_id = beliefs[0].compressed_belief_id
    finally:
        store.close()

    assert (
        main(
            [
                "decide-portfolio",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--compressed-belief-id",
                compressed_belief_id,
                "--sizing-method",
                "signal_weighted",
                "--sizing-engine",
                "optimizer",
                "--capital-base",
                "5.0",
                "--gross-exposure-cap",
                "0.8",
            ]
        )
        == 0
    )
    decision_output = capsys.readouterr().out
    assert "alpha-os portfolio decisions" in decision_output
    assert "subject=BTC_spot" in decision_output

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-signal-discovery-decision",
                    "--db",
                    str(db_path),
                    "--start-date",
                    "2026-03-24",
                    "--end-date",
                    "2026-03-24",
                    "--signal-discovery-id",
                    "core_crypto_search",
                    "--base-url",
                    "http://example.com",
                    "--portfolio-id",
                    "paper_core_workflow",
                    "--strategy-id",
                    "strategy:core_crypto_rule",
                    "--capital-base",
                    "5.0",
                    "--gross-exposure-cap",
                    "0.8",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader
    workflow_decision_output = capsys.readouterr().out
    assert "alpha-os workflow output" in workflow_decision_output
    assert "CompressedBeliefId:" in workflow_decision_output
    assert "alpha-os portfolio decisions" in workflow_decision_output
    assert "Strategy: strategy:core_crypto_rule" in workflow_decision_output
    assert "sizing=signal_weighted engine=rule_based" in workflow_decision_output
    assert "rebalance=every_1_steps" in workflow_decision_output
    assert "portfolio=paper_core_workflow" in workflow_decision_output
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        signal_discovery_runs = store.list_signal_discovery_runs(
            signal_discovery_id="core_crypto_search",
            execution_start_date="2026-03-23",
            execution_end_date="2026-03-24",
            limit=1,
        )
        assert len(signal_discovery_runs) == 1
        archived_snapshots = store.list_prepared_evaluation_snapshots(
            snapshot_set_id=signal_discovery_runs[0].run.snapshot_set_id,
        )
        assert len(archived_snapshots) > len(store.list_evaluation_snapshots(limit=10))
        workflow_decisions = store.list_portfolio_decisions(
            portfolio_id="paper_core_workflow",
            limit=10,
        )
        assert workflow_decisions
        workflow_details = workflow_decisions[0].details
        assert workflow_details is not None
        assert workflow_details["sizing_method"] == "signal_weighted"
        assert workflow_details["sizing_engine"] == "rule_based"
        assert workflow_details["strategy"]["strategy_id"] == "strategy:core_crypto_rule"
        assert workflow_details["strategy"]["rebalance"] == "every_1_steps"
    finally:
        store.close()

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "run-walk-forward",
                    "--db",
                    str(db_path),
                    "--evaluation-spec-id",
                    "core_crypto_eval",
                ]
            )
            == 0
        )
    finally:
        data_repositories.load_observation_frame = original_loader
    evaluation_output = capsys.readouterr().out
    assert "alpha-os evaluation report" in evaluation_output
    assert "Evaluation spec:  core_crypto_eval" in evaluation_output
    assert "CrossInstrumentReportContract:" in evaluation_output
    assert "contract=strategy,subject_set,universe_policy,instrument_mix,selection,sizing,rebalance,risk_caps,costs outcomes=metric_group_outcomes,failure_finding_outcomes" in evaluation_output
    assert "selection=all_assets" in evaluation_output
    assert "sizing=signal_weighted" in evaluation_output
    assert "rebalance=every_1_steps" in evaluation_output
    assert "subject_set=core_crypto summary=[bindings=1" in evaluation_output
    assert "metric_group_name=signal_discovery_result_stats" in evaluation_output
    assert "metric_group_name=signed_belief_quality" in evaluation_output
    assert "metric_group_name=portfolio_target_return_alignment" in evaluation_output
    assert "metric_group_name=sizing_policy_quality" in evaluation_output
    assert "metric_group_name=rebalance_policy_quality" in evaluation_output
    assert "metric_group_name=decision_quality" in evaluation_output
    assert "metric_group_name=robustness" in evaluation_output
    assert "source=native_plan" in evaluation_output
    assert "failure_metric_group=decision_quality" in evaluation_output
    assert "failure_metric_group=sizing_policy_quality" in evaluation_output
    assert "failure_metric_group=rebalance_policy_quality" in evaluation_output
    assert "failure_metric_group=signed_belief_quality" in evaluation_output
    assert "failure_metric_group=portfolio_target_return_alignment" in evaluation_output
    assert "evaluation_range_labels=eval_window" in evaluation_output
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        task_result = report_state.report.task_results[0]
        assert task_result.cross_instrument_outcome is not None
        outcome_dimensions = {
            item.metric_group_name for item in task_result.cross_instrument_outcome.metric_group_outcomes
        }
        assert "decision_quality" in outcome_dimensions
        failure_metric_groups = {
            item.metric_group_name for item in task_result.cross_instrument_outcome.failure_finding_outcomes
        }
        assert "decision_quality" in failure_metric_groups
        decision_metric_group_result = next(
            item for item in task_result.metric_group_results if item.metric_group_name == "decision_quality"
        )
        assert decision_metric_group_result.metrics["mean_decision_step_count"] > 1.0
    finally:
        store.close()


def test_core_crypto_4_end_to_end_runtime_smoke(tmp_path, capsys):
    import alpha_os.data_repositories as data_repositories
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"
    load_count = 0

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        nonlocal load_count
        import pandas as pd

        load_count += 1
        frames = {
            "BTC": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                        "2026-03-26T00:00:00Z",
                        "2026-03-27T00:00:00Z",
                    ],
                    "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 107.0, 106.0],
                }
            ),
            "ETH": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                        "2026-03-26T00:00:00Z",
                        "2026-03-27T00:00:00Z",
                    ],
                    "value": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0, 16.0, 15.0],
                }
            ),
            "SOL": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                        "2026-03-26T00:00:00Z",
                        "2026-03-27T00:00:00Z",
                    ],
                    "value": [20.0, 22.0, 21.0, 23.0, 25.0, 24.0, 26.0, 27.0],
                }
            ),
            "BNB": pd.DataFrame(
                {
                    "timestamp": [
                        "2026-03-20T00:00:00Z",
                        "2026-03-21T00:00:00Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-23T00:00:00Z",
                        "2026-03-24T00:00:00Z",
                        "2026-03-25T00:00:00Z",
                        "2026-03-26T00:00:00Z",
                        "2026-03-27T00:00:00Z",
                    ],
                    "value": [30.0, 31.0, 32.0, 31.0, 33.0, 34.0, 35.0, 36.0],
                }
            ),
        }
        return frames[asset]

    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto_4",
                "--observation-spec",
                "btc_close=daily_close",
                "--observation-spec",
                "eth_close=daily_close",
                "--observation-spec",
                "sol_close=daily_close",
                "--observation-spec",
                "bnb_close=daily_close",
                "--subject-binding",
                "BTC_spot=BTC=btc_close",
                "--subject-binding",
                "ETH_spot=ETH=eth_close",
                "--subject-binding",
                "SOL_spot=SOL=sol_close",
                "--subject-binding",
                "BNB_spot=BNB=bnb_close",
            ]
        )
        == 0
    )
    capsys.readouterr()

    original_loader = data_repositories.load_observation_frame
    data_repositories.load_observation_frame = _fake_loader
    try:
        assert (
            main(
                [
                    "debug-backfill-subject-set",
                    "--db",
                    str(db_path),
                    "--start-date",
                    "2026-03-23",
                    "--end-date",
                    "2026-03-24",
                    "--subject-set-id",
                    "core_crypto_4",
                    "--signal-candidate-spec-id",
                    "reversal_1d",
                    "--signal-candidate-spec-id",
                    "average_gap_3d",
                    "--base-url",
                    "http://example.com",
                ]
            )
            == 0
        )
        backfill_output = capsys.readouterr().out
        assert "subject_set=core_crypto_4" in backfill_output
        assert "executables=8" in backfill_output
        assert load_count == 4

        assert (
            main(
                [
                    "debug-compare-meta-aggregations",
                    "--db",
                    str(db_path),
                    "--subject-set-id",
                    "core_crypto_4",
                    "--target-id",
                    "residual_return_3d",
                ]
            )
            == 0
        )
        compare_output = capsys.readouterr().out
        assert "BTC / residual_return_3d" in compare_output
        assert "ETH / residual_return_3d" in compare_output
        assert "SOL / residual_return_3d" in compare_output
        assert "BNB / residual_return_3d" in compare_output

        assert (
            main(
                [
                    "debug-decide-portfolio-runtime",
                    "--db",
                    str(db_path),
                    "--portfolio-id",
                    "paper_core",
                    "--subject-set-id",
                    "core_crypto_4",
                    "--sizing-method",
                    "signal_weighted",
                    "--sizing-engine",
                    "optimizer",
                    "--capital-base",
                    "10",
                    "--gross-exposure-cap",
                    "0.8",
                    "--gross-limit",
                    "1.0",
                    "--net-limit",
                    "0.6",
                ]
            )
            == 0
        )
        build_output = capsys.readouterr().out
        assert "subject=BTC_spot" in build_output
        assert "subject=ETH_spot" in build_output
        assert "subject=SOL_spot" in build_output
        assert "subject=BNB_spot" in build_output
    finally:
        data_repositories.load_observation_frame = original_loader

    assert (
        main(
            [
                "debug-show-portfolio-decisions",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--limit",
                "20",
            ]
        )
        == 0
    )
    decisions_output = capsys.readouterr().out
    assert "Count:    4" in decisions_output
    assert "subject=BTC_spot" in decisions_output
    assert "subject=ETH_spot" in decisions_output
    assert "subject=SOL_spot" in decisions_output
    assert "subject=BNB_spot" in decisions_output


def test_subject_set_cli_registers_and_reuses_named_subject_set(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_meta_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-26",
            asset="BTC",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.2,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.2},{"prediction":0.1}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="BTC",
            target_id="residual_return_3d",
            corr=0.3,
            sample_count=10,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        store.upsert_meta_prediction(
            evaluation_id="ETH:residual_return_3d:2026-03-26",
            asset="ETH",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.1,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.1},{"prediction":0.05}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="ETH",
            target_id="residual_return_3d",
            corr=0.2,
            sample_count=8,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        for date, btc_obs, eth_obs in (
            ("2026-03-24", 0.10, 0.20),
            ("2026-03-25", 0.20, 0.40),
            ("2026-03-26", 0.05, 0.10),
        ):
            store.finalize_observation(
                evaluation_id=f"BTC:residual_return_3d:{date}",
                observation_value=btc_obs,
                asset="BTC",
                target_id="residual_return_3d",
            )
            store.finalize_observation(
                evaluation_id=f"ETH:residual_return_3d:{date}",
                observation_value=eth_obs,
                asset="ETH",
                target_id="residual_return_3d",
            )
    finally:
        store.close()

    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto",
                "--observation-spec",
                "btc_close=daily_close",
                "--observation-spec",
                "eth_close=daily_close",
                "--subject-binding",
                "BTC_spot=BTC=btc_close",
                "--subject-binding",
                "ETH_spot=ETH=eth_close",
            ]
        )
        == 0
    )
    register_output = capsys.readouterr().out
    assert "alpha-os subject sets" in register_output
    assert "core_crypto" in register_output

    assert (
        main(
            [
                "debug-show-subject-sets",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    show_sets_output = capsys.readouterr().out
    assert "core_crypto" in show_sets_output
    assert "btc_close=daily_close@signal_noise_asset_observable" in show_sets_output
    assert "eth_close=daily_close@signal_noise_asset_observable" in show_sets_output
    assert "BTC_spot=asset=BTC=btc_close" in show_sets_output
    assert "ETH_spot=asset=ETH=eth_close" in show_sets_output

    assert (
        main(
                [
                "debug-decide-portfolio-runtime",
                "--db",
                str(db_path),
                "--portfolio-id",
                "paper_core",
                "--subject-set-id",
                "core_crypto",
                "--sizing-method",
                "signal_weighted",
                "--sizing-engine",
                "optimizer",
                "--gross-exposure-cap",
                "0.5",
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert "alpha-os portfolio decisions" in build_output
    assert "subject=BTC_spot" in build_output
    assert "subject=ETH_spot" in build_output


def test_apply_runtime_manifest_accepts_subject_set_instruments(tmp_path, capsys):
    import json

    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "subject_sets": [
                    {
                        "subject_set_id": "macro_futures",
                        "instruments": [
                            {
                                "instrument_id": "es_front",
                                "instrument_type": "future",
                                "asset": "ES",
                                "venue": "CME",
                                "quote_ccy": "USD",
                                "contract_family": "ES",
                                "asset_class": "equity_index",
                                "region": "us",
                                "liquidity_tier": "tier1",
                                "cluster": "eq_index_dm",
                                "roll_rule": "volume_switch",
                                "multiplier": 50.0,
                            }
                        ],
                        "observation_specs": [
                            {
                                "observation_spec_id": "es_close",
                                "observable_id": "daily_close",
                            }
                        ],
                        "bindings": [
                            {
                                "subject_id": "ES_front",
                                "subject_kind": "future",
                                "asset": "ES",
                                "observation_spec_id": "es_close",
                                "instrument_id": "es_front",
                            }
                        ],
                    }
                ]
            }
        )
    )

    assert main(["apply-manifest", "--db", str(db_path), "--manifest", str(manifest_path)]) == 0
    output = capsys.readouterr().out
    assert "SubjectSets:    total=1 upserted=1" in output

    store = EvaluationStore(db_path)
    try:
        state = store.get_subject_set("macro_futures")
        assert state is not None
        definition = state.definition
        assert definition.instrument_id_by_subject == {"ES_front": "es_front"}
        instrument = definition.instrument_for_subject("ES_front")
        assert instrument is not None
        assert instrument.instrument_type == "future"
        assert instrument.venue == "CME"
        assert instrument.asset_class == "equity_index"
        assert instrument.region == "us"
        assert instrument.liquidity_tier == "tier1"
        assert instrument.cluster == "eq_index_dm"
        assert definition.subjects_grouped_by_instrument_field("cluster") == {
            "eq_index_dm": ("ES_front",)
        }
    finally:
        store.close()


def test_subject_set_cli_accepts_explicit_subject_kinds(tmp_path, capsys):
    from alpha_os.cli import main

    db_path = tmp_path / "runtime.db"

    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "macro_mix",
                "--observation-spec",
                "spy_close=daily_close",
                "--observation-spec",
                "vix_close=daily_close",
                "--subject-binding",
                "SPY_spot=equity=SPY=spy_close",
                "--subject-binding",
                "VIX_index=index=VIX=vix_close",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "debug-show-subject-sets",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    show_sets_output = capsys.readouterr().out
    assert "SPY_spot=equity=SPY=spy_close" in show_sets_output
    assert "VIX_index=index=VIX=vix_close" in show_sets_output


def test_check_subject_set_backend_cli_reports_availability(
    tmp_path, capsys, monkeypatch
):
    from alpha_os import cli as cli_module
    from alpha_os.cli import main

    class _FakeClient:
        base_url = "https://example.test"

        def resolve_observation(
            self,
            *,
            asset: str,
            observable_id: str,
            resolution: str = "1d",
            source_id: str = "signal_noise",
        ):
            if asset == "BTC":
                return {
                    "asset": asset,
                    "observable_id": observable_id,
                    "resolution": resolution,
                    "source_id": source_id,
                    "available": True,
                    "category": "crypto",
                    "signal_type": "ohlcv",
                    "last_updated": "2026-04-02T00:00:00Z",
                }
            return {
                "asset": asset,
                "observable_id": observable_id,
                "resolution": resolution,
                "source_id": source_id,
                "available": False,
                "category": None,
                "signal_type": None,
                "last_updated": None,
            }

    monkeypatch.setattr(
        cli_module,
        "build_signal_client",
        lambda *, base_url: _FakeClient(),
    )

    db_path = tmp_path / "runtime.db"
    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_check",
                "--observation-spec",
                "btc_close=daily_close",
                "--observation-spec",
                "spy_close=daily_close",
                "--subject-binding",
                "BTC_spot=asset=BTC=btc_close",
                "--subject-binding",
                "SPY_spot=equity=SPY=spy_close",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "check-subject-set-backend",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_check",
                "--base-url",
                "https://example.test",
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "alpha-os subject-set backend check" in output
    assert "BTC_spot kind=asset asset=BTC observable=daily_close source=signal_noise resolution=1d status=ok" in output
    assert "SPY_spot kind=equity asset=SPY observable=daily_close source=signal_noise resolution=1d status=missing" in output


def test_subject_set_runtime_views_aggregate_assets(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto",
                "--observation-spec",
                "btc_close=daily_close",
                "--observation-spec",
                "eth_close=daily_close",
                "--subject-binding",
                "BTC_spot=BTC=btc_close",
                "--subject-binding",
                "ETH_spot=ETH=eth_close",
            ]
        )
        == 0
    )
    capsys.readouterr()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_meta_prediction(
            evaluation_id="BTC:residual_return_3d:2026-03-26",
            asset="BTC",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.2,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.2}]}',
        )
        store.upsert_meta_prediction(
            evaluation_id="ETH:residual_return_3d:2026-03-26",
            asset="ETH",
            target_id="residual_return_3d",
            aggregation_kind="corr_weighted_mean",
            value=0.1,
            contributor_count=2,
            details_json='{"contributors":[{"prediction":0.1}]}',
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="BTC",
            target_id="residual_return_3d",
            corr=0.3,
            sample_count=10,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
        store.upsert_meta_prediction_metric(
            aggregation_kind="corr_weighted_mean",
            asset="ETH",
            target_id="residual_return_3d",
            corr=0.2,
            sample_count=8,
            window_size=20,
            start_evaluation_id=None,
            end_evaluation_id=None,
        )
    finally:
        store.close()

    assert (
        main(
            [
                "inspect-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto",
                "--evaluation-limit",
                "10",
                "--prediction-limit",
                "10",
            ]
        )
        == 0
    )
    inspect_output = capsys.readouterr().out
    assert "alpha-os subject-set inspection" in inspect_output
    assert "SubjectSet: core_crypto" in inspect_output
    assert "Assets:   BTC, ETH" in inspect_output
    assert "Evaluations:" in inspect_output
    assert "Meta:" in inspect_output
    assert "asset=BTC" in inspect_output
    assert "asset=ETH" in inspect_output

    assert (
        main(
            [
                "debug-compare-meta-aggregations",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto",
            ]
        )
        == 0
    )
    compare_output = capsys.readouterr().out
    assert "BTC / residual_return_3d" in compare_output
    assert "ETH / residual_return_3d" in compare_output


def test_validation_result_set_prints_subject_set_scope(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.signal_discovery import SignalDiscoverySpec
    from alpha_os.store import EvaluationStore
    from alpha_os.trading_strategy import (
        ExecutionPolicySpec,
        StrategyPortfolioSpec,
        TradingStrategyScopeSpec,
        TradingStrategySpec,
        RebalanceFrictionPolicySpec,
        HoldingCostPolicySpec,
    )
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )
    import alpha_os.validation_service as validation_service
    from alpha_os.validation_spec import ValidationDateRange, ValidationSpec, write_validation_spec

    db_path = tmp_path / "runtime.db"
    spec_path = tmp_path / "validation.json"
    assert (
        main(
            [
                "debug-register-subject-set",
                "--db",
                str(db_path),
                "--subject-set-id",
                "core_crypto",
                "--observation-spec",
                "btc_close=daily_close",
                "--subject-binding",
                "BTC_spot=BTC=btc_close",
            ]
        )
        == 0
    )
    capsys.readouterr()
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.upsert_signal_discovery_spec(
            "core_crypto_search",
            definition=SignalDiscoverySpec(
                signal_discovery_id="core_crypto_search",
                subject_set_id="core_crypto",
                signal_spec_ids=("momentum_1d",),
                target_id="residual_return_3d",
            ),
        )
        spec = TradingStrategySpec(
            strategy_id="strategy:core_crypto_search",
            label="Core Crypto Discovery",
            scope=TradingStrategyScopeSpec(
                subject_set_id="core_crypto",
                target_id=None,
            ),
            signal_discovery_id="core_crypto_search",
            position_rule_id="signal_discovery",
            family_mix=None,
            portfolio=StrategyPortfolioSpec(
                portfolio_construction=PortfolioConstructionSpec(
                    sizing_policy=PortfolioConstructionSizingSpec(
                        sizing_method="equal_weight",
                    ),
                    direction_mode=None,
                    gross_exposure_cap=None,
                ),
                rebalance_friction_policy=RebalanceFrictionPolicySpec(
                    turnover_friction=None,
                    no_trade_band=None,
                ),
                execution_policy=ExecutionPolicySpec(
                    market_impact_bps=None,
                    fee_bps=None,
                    bid_ask_spread_bps=None,
                ),
                holding_cost_policy=HoldingCostPolicySpec(),
                selection_kind="all_assets",
                top_k=None,
            ),
            created_at="2026-03-29T00:00:00+00:00",
        )
        store.upsert_trading_strategy(trading_strategy=spec)
    finally:
        store.close()
    write_validation_spec(
        spec_path,
        ValidationSpec(
            signal_ids=("momentum_1d", "reversal_1d"),
            target_ids=("residual_return_3d",),
            date_ranges=(
                ValidationDateRange(
                    label="recent_short",
                    start_date="2026-03-20",
                    end_date="2026-03-27",
                ),
            ),
            metric_windows=(20,),
            aggregation_kinds=("active_equal_mean",),
            subject_set_ids=("core_crypto",),
        ),
    )

    def _fake_loader(*, base_url: str, asset: str, observation_spec):
        import pandas as pd

        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-20T00:00:00Z",
                    "2026-03-21T00:00:00Z",
                    "2026-03-22T00:00:00Z",
                    "2026-03-23T00:00:00Z",
                    "2026-03-24T00:00:00Z",
                    "2026-03-25T00:00:00Z",
                    "2026-03-26T00:00:00Z",
                    "2026-03-27T00:00:00Z",
                ],
                "value": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 107.0, 106.0],
            }
        )

    original_loader = validation_service._load_price_frame_from_signal_noise
    validation_service._load_price_frame_from_signal_noise = _fake_loader
    try:
        assert (
            main(
                [
                    "validate-strategy",
                    "--db",
                    str(db_path),
                    "--strategy-id",
                    "strategy:core_crypto_search",
                    "--spec",
                    str(spec_path),
                    "--base-url",
                    "http://127.0.0.1:8000",
                ]
            )
            == 0
        )
    finally:
        validation_service._load_price_frame_from_signal_noise = original_loader
    capsys.readouterr()

    assert (
        main(
            [
                "debug-summarize-validation",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "subject_set=core_crypto" in output


def test_apply_evaluations_batch_persists_snapshot_artifacts(tmp_path):
    from alpha_os.evaluation_inputs import EvaluationInput
    from alpha_os.evaluation_runtime import apply_evaluations_batch
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "runtime.db"
    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        store.register_signal("hyp_a")

        latest_snapshot, created_count, existing_count = apply_evaluations_batch(
            store,
            evaluation_inputs=[
                EvaluationInput(
                    date="2026-03-27",
                    signal_id="hyp_a",
                    prediction=0.2,
                    observation=0.1,
                    funding_cost_bps=2.5,
                    borrow_fee_bps=3.5,
                    roll_cost_bps=4.5,
                    financing_cost_bps=1.25,
                    contract_multiplier=10.0,
                    contract_id="ESU2026",
                    contract_family="CME:ES",
                    quote_ccy="USD",
                    collateral_ccy="USD",
                    roll_event={
                        "contract_id": "ESU2026",
                        "rolled": True,
                        "roll_reason": "calendar_days_before_expiry",
                    },
                )
            ],
            input_source="test_batch",
            refresh_metrics=False,
        )
        assert created_count == 1
        assert existing_count == 0
        assert latest_snapshot is not None
        assert latest_snapshot.funding_cost_bps == pytest.approx(2.5)
        assert latest_snapshot.borrow_fee_bps == pytest.approx(3.5)
        assert latest_snapshot.roll_cost_bps == pytest.approx(4.5)
        assert latest_snapshot.financing_cost_bps == pytest.approx(1.25)
        assert latest_snapshot.contract_multiplier == pytest.approx(10.0)
        assert latest_snapshot.contract_id == "ESU2026"
        assert latest_snapshot.contract_family == "CME:ES"
        assert latest_snapshot.quote_ccy == "USD"
        assert latest_snapshot.collateral_ccy == "USD"
        assert latest_snapshot.roll_event == {
            "contract_id": "ESU2026",
            "rolled": True,
            "roll_reason": "calendar_days_before_expiry",
        }

        archived = store.archive_prepared_evaluation_snapshots(
            snapshot_set_id="run-1",
            signal_ids=["hyp_a"],
        )
        assert archived == 1
        archived_snapshots = store.list_prepared_evaluation_snapshots(
            snapshot_set_id="run-1",
            signal_ids=["hyp_a"],
        )
        assert len(archived_snapshots) == 1
        assert archived_snapshots[0].funding_cost_bps == pytest.approx(2.5)
        assert archived_snapshots[0].borrow_fee_bps == pytest.approx(3.5)
        assert archived_snapshots[0].roll_cost_bps == pytest.approx(4.5)
        assert archived_snapshots[0].financing_cost_bps == pytest.approx(1.25)
        assert archived_snapshots[0].contract_multiplier == pytest.approx(10.0)
        assert archived_snapshots[0].contract_id == "ESU2026"
        assert archived_snapshots[0].contract_family == "CME:ES"
        assert archived_snapshots[0].quote_ccy == "USD"
        assert archived_snapshots[0].collateral_ccy == "USD"
        assert archived_snapshots[0].roll_event == {
            "contract_id": "ESU2026",
            "rolled": True,
            "roll_reason": "calendar_days_before_expiry",
        }
    finally:
        store.close()


def test_list_runtime_manifests_shows_reference_and_examples(capsys):
    from alpha_os.cli import main

    assert main(["list-runtime-manifests"]) == 0
    output = capsys.readouterr().out

    assert "alpha-os runtime manifests" in output
    assert "global_macro_futures_daily_trend.json category=reference instrument_types=future,perp subject_kinds=future,perp" in output
    assert "fixture_daily_diagnostic.json category=diagnostic instrument_types=- subject_kinds=equity" in output
    assert "global_macro_tradeable_daily_diagnostic.json category=diagnostic" in output
    assert "global_macro_tradeable_daily_10y.json category=cross_asset_example instrument_types=future,perp subject_kinds=future,perp" in output
    assert "us_equity_narrow_directional_context.json category=equity_example instrument_types=- subject_kinds=equity" in output
    assert "us_etf_standard_rs_lowvol_equal_weight.json category=etf_example instrument_types=- subject_kinds=etf" in output


def test_global_macro_diagnostic_manifest_contract():
    from pathlib import Path

    import json

    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "runtime_manifests"
        / "global_macro_tradeable_daily_diagnostic.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["extends_manifest"] == "global_macro_tradeable_daily_10y.json"
    evaluation_spec = manifest["evaluation_specs"][0]
    assert evaluation_spec["evaluation_spec_id"] == "global_macro_tradeable_daily_diagnostic_eval"
    assert len(evaluation_spec["evaluation_folds"]) == 1
    assert evaluation_spec["evaluation_folds"][0]["evaluation_date_ranges"][0]["label"] == "2024"
    assert {
        "prediction_diagnostics",
        "portfolio_construction_trace",
        "execution_trace",
        "cost_drag",
        "signal_churn",
        "portfolio_concentration",
        "sizing_policy_quality",
        "rebalance_policy_quality",
    } <= set(evaluation_spec["metric_group_names"])
    findings = manifest["evaluation_tasks"]
    assert len(findings) == 15
    case_ids = {item["evaluation_task_id"] for item in findings}
    assert {
        "global_macro_tradeable_daily_diagnostic_equal_weight_hold_case",
        "global_macro_tradeable_daily_diagnostic_equal_weight_monthly_hold_case",
        "global_macro_tradeable_daily_diagnostic_case",
        "global_macro_tradeable_daily_diagnostic_cost_aware_execution_case",
        "global_macro_tradeable_daily_diagnostic_utility_looser_benefit_case",
        "global_macro_tradeable_daily_diagnostic_utility_tighter_benefit_case",
        "global_macro_tradeable_daily_diagnostic_utility_looser_budget_case",
        "global_macro_tradeable_daily_diagnostic_utility_no_budget_case",
        "global_macro_tradeable_daily_diagnostic_mean_reversion_case",
        "global_macro_tradeable_daily_diagnostic_mean_reversion_constrained_case",
        "global_macro_tradeable_daily_diagnostic_mean_reversion_optimizer_case",
        "global_macro_tradeable_daily_diagnostic_legacy_proportional_execution_case",
        "global_macro_tradeable_daily_diagnostic_no_cost_case",
        "global_macro_tradeable_daily_diagnostic_weekly_rebalance_case",
        "global_macro_tradeable_daily_diagnostic_no_risk_budget_case",
    } == case_ids
    for case in findings:
        assert case["evaluation_spec_id"] == "global_macro_tradeable_daily_diagnostic_eval"
        strategy_override = case["strategy_override"]
        construction = strategy_override["portfolio_construction"]
        assert "portfolio_construction" not in case
        assert "rebalance_friction_policy" not in case
        if case["evaluation_task_id"] == "global_macro_tradeable_daily_diagnostic_equal_weight_hold_case":
            assert case["strategy_id"] == "strategy:global_macro_tradeable_daily_diagnostic_equal_weight_hold"
            assert construction["construction_kind"] == "hold_baseline"
            assert "signal_discovery_id" not in case
        elif case["evaluation_task_id"] == "global_macro_tradeable_daily_diagnostic_equal_weight_monthly_hold_case":
            assert case["strategy_id"] == "strategy:global_macro_tradeable_daily_diagnostic_equal_weight_hold"
            assert construction["construction_kind"] == "hold_baseline"
            assert strategy_override["rebalance_interval_steps"] == 21
            assert "signal_discovery_id" not in case
        elif case["evaluation_task_id"] in {
            "global_macro_tradeable_daily_diagnostic_case",
            "global_macro_tradeable_daily_diagnostic_cost_aware_execution_case",
            "global_macro_tradeable_daily_diagnostic_utility_tighter_benefit_case",
            "global_macro_tradeable_daily_diagnostic_utility_looser_budget_case",
            "global_macro_tradeable_daily_diagnostic_utility_no_budget_case",
            "global_macro_tradeable_daily_diagnostic_legacy_proportional_execution_case",
            "global_macro_tradeable_daily_diagnostic_no_cost_case",
            "global_macro_tradeable_daily_diagnostic_weekly_rebalance_case",
            "global_macro_tradeable_daily_diagnostic_mean_reversion_constrained_case",
            "global_macro_tradeable_daily_diagnostic_mean_reversion_optimizer_case",
        }:
            if (
                case["evaluation_task_id"]
                == "global_macro_tradeable_daily_diagnostic_mean_reversion_constrained_case"
            ):
                assert (
                    case["signal_discovery_id"]
                    == "global_macro_tradeable_daily_diagnostic_mean_reversion_search"
                )
                assert strategy_override["rebalance_interval_steps"] == 10
                assert construction["risk_budget"]["target_gross_exposure"] == 0.35
                assert construction["portfolio_intent"] == {
                    "effective_n_floor": 10.0,
                    "top_gross_share_cap_n": 3,
                    "top_gross_share_cap": 0.4,
                }
                assert (
                    construction["sleeve_composition"]["sleeves"][0][
                        "sleeve_kind"
                    ]
                    == "mean_reversion"
                )
                assert strategy_override["rebalance_friction_policy"] == {
                    "execution_mode": "utility_priority",
                    "turnover_friction": 0.001,
                    "no_trade_band": 0.01,
                    "execution_cost_aversion": 3.0,
                    "turnover_budget": 0.025,
                    "partial_fill_enabled": True,
                }
            if (
                case["evaluation_task_id"]
                == "global_macro_tradeable_daily_diagnostic_mean_reversion_optimizer_case"
            ):
                assert (
                    case["signal_discovery_id"]
                    == "global_macro_tradeable_daily_diagnostic_mean_reversion_search"
                )
                assert construction["sizing_policy"] == {
                    "sizing_method": "signed_mean_variance",
                    "sizing_engine": "optimizer",
                }
                assert strategy_override["rebalance_interval_steps"] == 21
                assert (
                    construction["sleeve_composition"]["sleeves"][0][
                        "sleeve_kind"
                    ]
                    == "mean_reversion"
                )
        elif (
            case["evaluation_task_id"]
            == "global_macro_tradeable_daily_diagnostic_mean_reversion_case"
        ):
            assert (
                case["signal_discovery_id"]
                == "global_macro_tradeable_daily_diagnostic_mean_reversion_search"
            )
            assert (
                construction["sleeve_composition"]["sleeves"][0][
                    "sleeve_kind"
                ]
                == "mean_reversion"
            )
        else:
            assert case["signal_discovery_id"] == "global_macro_tradeable_daily_diagnostic_search"
