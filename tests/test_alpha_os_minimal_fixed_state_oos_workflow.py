from alpha_os.cli import main
from alpha_os.evaluation_runner import EvaluationRunRequest, evaluate_evaluation_spec_state
from alpha_os.evaluation_task import EvaluationTask, build_evaluation_task_id
from alpha_os.store import EvaluationStore


def test_minimal_fixed_state_oos_golden_path_runs_without_external_services(
    tmp_path,
    capsys,
):
    db_path = tmp_path / "alpha-os-minimal-fixed-state-oos.db"

    assert (
        main(
            [
                "apply-manifest",
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
        strategy_checkpoints = store.list_strategy_checkpoints(
            signal_discovery_id="minimal_fixed_state_search",
            limit=10,
        )
        assert strategy_checkpoints
        source_state = strategy_checkpoints[0].state
        assert source_state.screening_result_id
        assert source_state.compressed_belief_id
        source_task_state = store.get_evaluation_task("minimal_fixed_state_training_case")
        assert source_task_state is not None
        source_task = source_task_state.task
        strict_spec_state = store.get_evaluation_spec("minimal_fixed_state_oos_eval")
        assert strict_spec_state is not None
        checkpoint_task = EvaluationTask(
            evaluation_task_id=build_evaluation_task_id(
                strategy_id=source_task.strategy_id,
                evaluation_spec_id=strict_spec_state.evaluation_spec_id,
            ),
            strategy_id=source_task.strategy_id,
            evaluation_spec_id=strict_spec_state.evaluation_spec_id,
        )
        evaluate_evaluation_spec_state(
            EvaluationRunRequest(
                store=store,
                evaluation_spec_state=strict_spec_state,
                evaluation_tasks=(checkpoint_task,),
                base_url="http://127.0.0.1:8000",
            )
        )
    finally:
        store.close()

    store = EvaluationStore(db_path)
    try:
        store.ensure_schema()
        strict_spec_state = store.get_evaluation_spec("minimal_fixed_state_oos_eval")
        assert strict_spec_state is not None
        strict_spec = strict_spec_state.definition
        assert strict_spec.rigor_level == "fixed_state_oos"
        assert strict_spec.oos_contract.enforcement == "strict"
        assert strict_spec.oos_contract.require_strategy_checkpoint_for_trained_strategy is True

        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        report = report_state.report
        assert report.evaluation_spec_id == "minimal_fixed_state_oos_eval"
        assert report.oos_contract_summary["rigor_level"] == "fixed_state_oos"
        assert report.oos_contract_summary["enforcement"] == "strict"
        assert report.oos_contract_summary["range_non_overlap"] == "pass"
        assert report.oos_contract_summary["evaluation_after_execution"] == "pass"
        assert report.oos_contract_summary["strategy_checkpoint_required"] == "required"
        assert len(report.task_results) == 1

        task_result = report.task_results[0]
        assert task_result.artifact_refs["strategy_checkpoint_ids"] == (
            source_state.strategy_checkpoint_id,
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
