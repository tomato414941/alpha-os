from alpha_os.cli import main
from alpha_os.store import EvaluationStore


def test_minimal_fixed_state_oos_golden_path_runs_without_external_services(
    tmp_path,
    capsys,
):
    db_path = tmp_path / "alpha-os-minimal-fixed-state-oos.db"

    assert (
        main(
            [
                "apply-runtime-manifest",
                "--manifest",
                "examples/minimal_fixed_state_oos.json",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "run-walk-forward-evaluation",
                "--evaluation-spec-id",
                "minimal_fixed_state_train_eval",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        initial_strategy_states = store.list_initial_strategy_states(
            signal_discovery_id="minimal_fixed_state_search",
            limit=10,
        )
        assert initial_strategy_states
        source_state = initial_strategy_states[0].state
        assert source_state.signal_train_id
        assert source_state.signal_discovery_run_id
        assert source_state.screening_result_id
        assert source_state.compressed_belief_id
    finally:
        store.close()

    assert (
        main(
            [
                "create-fixed-state-evaluation-task",
                "--source-evaluation-task-id",
                "minimal_fixed_state_training_case",
                "--source-initial-strategy-state-id",
                source_state.initial_strategy_state_id,
                "--evaluation-spec-id",
                "minimal_fixed_state_oos_eval",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "run-walk-forward-evaluation",
                "--evaluation-spec-id",
                "minimal_fixed_state_oos_eval",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["show-evaluation-report", "--db", str(db_path)]) == 0
    report_output = capsys.readouterr().out
    assert "OOS contract: rigor_level=fixed_state_oos enforcement=strict" in report_output
    assert "range_non_overlap=pass" in report_output
    assert "evaluation_after_execution=pass" in report_output
    assert "frozen_state_required=required" in report_output

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        strict_spec_state = store.get_evaluation_spec("minimal_fixed_state_oos_eval")
        assert strict_spec_state is not None
        strict_spec = strict_spec_state.definition
        assert strict_spec.rigor_level == "fixed_state_oos"
        assert strict_spec.oos_contract.enforcement == "strict"
        assert strict_spec.oos_contract.require_frozen_state_for_trained_strategy is True

        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        report = report_state.report
        assert report.evaluation_spec_id == "minimal_fixed_state_oos_eval"
        assert report.oos_contract_summary["rigor_level"] == "fixed_state_oos"
        assert report.oos_contract_summary["enforcement"] == "strict"
        assert report.oos_contract_summary["range_non_overlap"] == "pass"
        assert report.oos_contract_summary["evaluation_after_execution"] == "pass"
        assert report.oos_contract_summary["frozen_state_required"] == "required"
        assert len(report.task_results) == 1

        task_result = report.task_results[0]
        assert task_result.artifact_refs["initial_strategy_state_ids"] == (
            source_state.initial_strategy_state_id,
        )
        assert task_result.artifact_refs["signal_train_ids"] == (
            source_state.signal_train_id,
        )
        assert task_result.artifact_refs["signal_discovery_run_ids"] == (
            source_state.signal_discovery_run_id,
        )
        assert task_result.artifact_refs["screening_result_ids"] == (
            source_state.screening_result_id,
        )
        assert task_result.artifact_refs["compressed_belief_ids"] == (
            source_state.compressed_belief_id,
        )
        assert task_result.artifact_refs["evaluation_fold_labels"] == (
            "minimal_fixed_replay_fold",
        )
        assert task_result.artifact_refs["evaluation_range_labels"] == (
            "minimal_fixed_strict_oos",
        )
        assert task_result.strategy_contract_fields["subject_set"] == (
            "minimal_fixed_state_pair"
        )
        assert task_result.strategy_contract_fields["target_id"] == "residual_return_3d"
    finally:
        store.close()
