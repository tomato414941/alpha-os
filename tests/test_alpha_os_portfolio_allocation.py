import pytest

from alpha_os.portfolio_allocation import EqualWeightLongOnlyAllocator


def test_equal_weight_long_only_allocator_splits_long_candidates() -> None:
    allocation = EqualWeightLongOnlyAllocator().allocate(
        {
            "BTC": "long",
            "ETH": "long",
            "SOL": "flat",
        }
    )

    assert allocation == {
        "BTC": 0.5,
        "ETH": 0.5,
        "SOL": 0.0,
    }


def test_equal_weight_long_only_allocator_ignores_short_candidates() -> None:
    allocation = EqualWeightLongOnlyAllocator(gross_exposure_cap=0.8).allocate(
        {
            "BTC": "long",
            "ETH": "short",
        }
    )

    assert allocation == {
        "BTC": 0.8,
        "ETH": 0.0,
    }


def test_equal_weight_long_only_allocator_returns_flat_when_no_long_candidates() -> None:
    allocation = EqualWeightLongOnlyAllocator().allocate(
        {
            "BTC": "flat",
            "ETH": "short",
        }
    )

    assert allocation == {
        "BTC": 0.0,
        "ETH": 0.0,
    }


def test_equal_weight_long_only_allocator_rejects_negative_gross_exposure() -> None:
    with pytest.raises(ValueError, match="gross_exposure_cap must be non-negative"):
        EqualWeightLongOnlyAllocator(gross_exposure_cap=-1.0)
