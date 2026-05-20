from pathlib import Path


def test_minimal_oos_golden_path_runs_without_external_services(tmp_path, capsys):
    from alpha_os.cli import main
    from alpha_os.store import EvaluationStore

    db_path = tmp_path / "alpha-os.db"
    manifest_path = Path(__file__).resolve().parents[1] / "examples" / "minimal_oos.json"

    assert (
        main(
            [
                "apply-manifest",
                "--manifest",
                str(manifest_path),
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    apply_output = capsys.readouterr().out
    assert "minimal_oos_eval" in apply_output
    assert "minimal_oos_pair" in apply_output

    assert (
        main(
            [
                "run-walk-forward-evaluation",
                "--evaluation-spec-id",
                "minimal_oos_eval",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    run_output = capsys.readouterr().out
    assert "alpha-os evaluation run" in run_output
    assert "Evaluation spec:  minimal_oos_eval" in run_output
    assert "TaskResults: 2" in run_output

    store = EvaluationStore(db_path)
    try:
        evaluation_spec_state = store.get_evaluation_spec("minimal_oos_eval")
        assert evaluation_spec_state is not None
        evaluation_spec = evaluation_spec_state.definition
        assert evaluation_spec.rigor_level == "diagnostic"
        assert evaluation_spec.oos_contract.enforcement == "warn"
        assert evaluation_spec.execution_range.start_date == "2026-01-01"
        assert evaluation_spec.execution_range.end_date == "2026-01-15"
        assert evaluation_spec.evaluation_folds[0].label == "minimal_train_to_test"
        assert (
            evaluation_spec.evaluation_folds[0].evaluation_date_ranges[0].start_date
            == "2026-01-16"
        )
        assert (
            evaluation_spec.evaluation_folds[0].evaluation_date_ranges[0].end_date
            == "2026-01-31"
        )

        candidate_strategy = store.get_trading_strategy(
            "strategy:minimal_oos_candidate_equal_weight_hold"
        )
        assert candidate_strategy is not None
        assert candidate_strategy.trading_strategy.scope.subject_set_id == "minimal_oos_pair"
        assert candidate_strategy.trading_strategy.scope.target_id == "residual_return_1d"

        report_state = store.get_latest_evaluation_report()
        assert report_state is not None
        report = report_state.report
        assert report.evaluation_spec_id == "minimal_oos_eval"
        assert report.oos_contract_summary["rigor_level"] == "diagnostic"
        assert report.oos_contract_summary["enforcement"] == "warn"
        assert report.oos_contract_summary["range_non_overlap"] == "pass"
        assert report.oos_contract_summary["evaluation_after_execution"] == "pass"
        assert len(report.task_results) == 2
        task_results = {item.evaluation_task_id: item for item in report.task_results}
        assert set(task_results) == {
            "minimal_oos_candidate_equal_weight_hold_case",
            "minimal_oos_baseline_equal_weight_hold_case",
        }

        candidate_result = task_results["minimal_oos_candidate_equal_weight_hold_case"]
        assert candidate_result.strategy_id == (
            "strategy:minimal_oos_candidate_equal_weight_hold"
        )
    finally:
        store.close()
