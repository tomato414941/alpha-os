import pytest

from alpha_os.cli import main


def test_minimal_fixed_state_oos_requires_prepared_strategy_checkpoint(
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

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "run-walk-forward-evaluation",
                "--evaluation-spec-id",
                "minimal_fixed_state_train_eval",
                "--db",
                str(db_path),
            ]
        )

    assert exc_info.value.code == 2
    assert "requires a strategy checkpoint" in capsys.readouterr().err
