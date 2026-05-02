import pytest

from alpha_os.portfolio_allocation import (
    EqualWeightLongOnlyAllocator,
    PositionCandidate,
)


def test_equal_weight_long_only_allocator_splits_long_candidates() -> None:
    allocation = EqualWeightLongOnlyAllocator().allocate(
        (
            PositionCandidate(subject_id="BTC", direction="long"),
            PositionCandidate(subject_id="ETH", direction="long"),
            PositionCandidate(subject_id="SOL", direction="flat"),
        )
    )

    assert allocation.target_weights == {
        "BTC": 0.5,
        "ETH": 0.5,
        "SOL": 0.0,
    }


def test_equal_weight_long_only_allocator_ignores_short_candidates() -> None:
    allocation = EqualWeightLongOnlyAllocator(gross_exposure_cap=0.8).allocate(
        (
            PositionCandidate(subject_id="BTC", direction="long"),
            PositionCandidate(subject_id="ETH", direction="short"),
        )
    )

    assert allocation.target_weights == {
        "BTC": 0.8,
        "ETH": 0.0,
    }


def test_equal_weight_long_only_allocator_returns_flat_when_no_long_candidates() -> None:
    allocation = EqualWeightLongOnlyAllocator().allocate(
        (
            PositionCandidate(subject_id="BTC", direction="flat"),
            PositionCandidate(subject_id="ETH", direction="short"),
        )
    )

    assert allocation.target_weights == {
        "BTC": 0.0,
        "ETH": 0.0,
    }


def test_equal_weight_long_only_allocator_rejects_negative_gross_exposure() -> None:
    with pytest.raises(ValueError, match="gross_exposure_cap must be non-negative"):
        EqualWeightLongOnlyAllocator(gross_exposure_cap=-1.0)

