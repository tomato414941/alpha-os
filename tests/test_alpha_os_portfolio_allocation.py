from __future__ import annotations

from alpha_os.portfolio_allocation import (
    EqualWeightLongOnlyAllocator,
    PositionCandidate,
)


def test_equal_weight_long_only_allocator_weights_long_candidates() -> None:
    allocation = EqualWeightLongOnlyAllocator(gross_exposure_cap=1.0).allocate(
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


def test_equal_weight_long_only_allocator_respects_gross_exposure_cap() -> None:
    allocation = EqualWeightLongOnlyAllocator(gross_exposure_cap=0.6).allocate(
        (
            PositionCandidate(subject_id="BTC", direction="long"),
            PositionCandidate(subject_id="ETH", direction="long"),
            PositionCandidate(subject_id="SOL", direction="flat"),
        )
    )

    assert allocation.target_weights == {
        "BTC": 0.3,
        "ETH": 0.3,
        "SOL": 0.0,
    }


def test_equal_weight_long_only_allocator_flats_everything_without_longs() -> None:
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

