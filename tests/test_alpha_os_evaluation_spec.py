import pytest


def test_evaluation_spec_rejects_invalid_date():
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec

    with pytest.raises(
        ValueError,
        match="evaluation date range train has invalid start_date: 2025-02-30",
    ):
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-02-30",
                end_date="2025-03-01",
            )
        )


def test_evaluation_spec_rejects_non_positive_metric_windows():
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec

    with pytest.raises(
        ValueError,
        match="evaluation spec metric_windows must be positive integers: 0",
    ):
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-01-01",
                end_date="2025-01-31",
            ),
            metric_windows=(0,),
        )


def test_evaluation_spec_rejects_unknown_aggregation_kinds():
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec

    with pytest.raises(
        ValueError,
        match="evaluation spec has unknown aggregation kinds: mystery",
    ):
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-01-01",
                end_date="2025-01-31",
            ),
            aggregation_kinds=("mystery",),
        )


def test_evaluation_spec_rejects_empty_target_ids():
    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec

    with pytest.raises(
        ValueError,
        match="evaluation spec target_ids must be non-empty strings",
    ):
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-01-01",
                end_date="2025-01-31",
            ),
            target_ids=("",),
        )


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


def test_evaluation_spec_rejects_overlapping_train_test_in_strict_oos_contract():
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationOosContract,
        EvaluationSpec,
    )

    with pytest.raises(
        ValueError,
        match="evaluation OOS contract violation: execution and evaluation ranges overlap",
    ):
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-01-01",
                end_date="2025-01-31",
            ),
            evaluation_date_ranges=(
                EvaluationDateRange(
                    label="test",
                    start_date="2025-01-15",
                    end_date="2025-02-15",
                ),
            ),
            rigor_level="backtest_oos",
            oos_contract=EvaluationOosContract(enforcement="strict"),
        )


def test_evaluation_spec_warns_on_overlapping_train_test_in_diagnostic_contract():
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationOosContract,
        EvaluationSpec,
    )

    with pytest.warns(
        UserWarning,
        match="evaluation OOS contract violation: execution and evaluation ranges overlap",
    ):
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-01-01",
                end_date="2025-01-31",
            ),
            evaluation_date_ranges=(
                EvaluationDateRange(
                    label="test",
                    start_date="2025-01-15",
                    end_date="2025-02-15",
                ),
            ),
            rigor_level="diagnostic",
            oos_contract=EvaluationOosContract(
                enforcement="warn",
                require_evaluation_after_execution=False,
            ),
        )


def test_evaluation_spec_does_not_warn_on_exploratory_overlap_by_default():
    import warnings

    from alpha_os.evaluation_spec import EvaluationDateRange, EvaluationSpec

    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-01-01",
                end_date="2025-01-31",
            ),
            evaluation_date_ranges=(
                EvaluationDateRange(
                    label="test",
                    start_date="2025-01-15",
                    end_date="2025-02-15",
                ),
            ),
        )

    assert not warning_records


def test_evaluation_spec_rejects_evaluation_before_execution_in_strict_contract():
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationOosContract,
        EvaluationSpec,
    )

    with pytest.raises(
        ValueError,
        match=(
            "evaluation OOS contract violation: evaluation range does not start "
            "after execution range"
        ),
    ):
        EvaluationSpec(
            execution_range=EvaluationDateRange(
                label="train",
                start_date="2025-02-01",
                end_date="2025-02-28",
            ),
            evaluation_date_ranges=(
                EvaluationDateRange(
                    label="test",
                    start_date="2025-01-01",
                    end_date="2025-01-31",
                ),
            ),
            rigor_level="backtest_oos",
            oos_contract=EvaluationOosContract(enforcement="strict"),
        )


def test_evaluation_spec_roundtrips_oos_contract_document():
    from alpha_os.evaluation_spec import (
        EvaluationDateRange,
        EvaluationOosContract,
        EvaluationSpec,
    )

    evaluation_spec = EvaluationSpec(
        execution_range=EvaluationDateRange(
            label="train",
            start_date="2025-01-01",
            end_date="2025-01-31",
        ),
        evaluation_date_ranges=(
            EvaluationDateRange(
                label="test",
                start_date="2025-02-01",
                end_date="2025-02-28",
            ),
        ),
        rigor_level="backtest_oos",
        oos_contract=EvaluationOosContract(enforcement="strict"),
    )

    roundtripped = EvaluationSpec.from_document(evaluation_spec.to_document())

    assert roundtripped.rigor_level == "backtest_oos"
    assert roundtripped.oos_contract == EvaluationOosContract(enforcement="strict")


def test_minimal_oos_manifest_remains_valid_as_diagnostic_warn():
    import json
    from pathlib import Path

    from alpha_os.evaluation_spec import EvaluationSpec

    manifest = json.loads(Path("examples/minimal_oos.json").read_text())
    document = dict(manifest["evaluation_specs"][0])
    document["rigor_level"] = "diagnostic"
    document["oos_contract"] = {"enforcement": "warn"}

    evaluation_spec = EvaluationSpec.from_document(document)

    assert evaluation_spec.rigor_level == "diagnostic"
    assert evaluation_spec.oos_contract.enforcement == "warn"


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
