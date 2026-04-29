import pytest


def test_evaluation_spec_rejects_reversed_date_ranges():
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec

    with pytest.raises(
        ValueError,
        match="evaluation date range has start_date after end_date: train",
    ):
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-01-02",
                end_date="2025-01-01",
            )
        )


def test_evaluation_spec_rejects_duplicate_fold_labels():
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationFold,
        EvaluationSpec,
    )

    execution_range = EvaluationDateRange(
        label="train",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    with pytest.raises(
        ValueError,
        match="evaluation spec contains duplicate fold labels: fold_a",
    ):
        EvaluationSpec(
            execution_range=execution_range,
            evaluation_folds=(
                EvaluationFold(label="fold_a", execution_range=execution_range),
                EvaluationFold(label="fold_a", execution_range=execution_range),
            ),
        )


def test_evaluation_spec_rejects_duplicate_fold_evaluation_range_labels():
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationFold,
        EvaluationSpec,
    )

    execution_range = EvaluationDateRange(
        label="train",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    evaluation_range = EvaluationDateRange(
        label="oos",
        start_date="2025-02-01",
        end_date="2025-02-28",
    )

    with pytest.raises(
        ValueError,
        match=(
            "evaluation spec contains duplicate evaluation range for fold fold_a "
            "labels: oos"
        ),
    ):
        EvaluationSpec(
            execution_range=execution_range,
            evaluation_folds=(
                EvaluationFold(
                    label="fold_a",
                    execution_range=execution_range,
                    evaluation_date_ranges=(evaluation_range, evaluation_range),
                ),
            ),
        )
