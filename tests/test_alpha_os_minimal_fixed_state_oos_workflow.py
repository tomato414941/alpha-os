from alpha_os.cli import main


def test_minimal_fixed_state_oos_training_eval_runs_without_checkpoint_preparation(
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

    output = capsys.readouterr().out
    assert "alpha-os evaluation run" in output
    assert "Evaluation spec:  minimal_fixed_state_train_eval" in output
    assert "TaskResults: 1" in output
