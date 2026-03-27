from __future__ import annotations

import pytest


def test_build_cycle_update_rewards_positive_edge():
    from alpha_os.policy import build_cycle_update

    update = build_cycle_update(
        quality_before=0.1,
        allocation_trust_before=0.2,
        prediction_value=0.5,
        observation_value=0.2,
    )

    assert update.signed_edge == pytest.approx(0.1)
    assert update.absolute_error == pytest.approx(0.3)
    assert update.quality_after == pytest.approx(0.1)
    assert update.quality_delta == pytest.approx(0.0)
    assert update.allocation_trust_after < update.allocation_trust_before
    assert update.allocation_trust_delta < 0.0


def test_build_cycle_update_reduces_trust_after_negative_edge():
    from alpha_os.policy import build_cycle_update

    update = build_cycle_update(
        quality_before=-0.2,
        allocation_trust_before=0.05,
        prediction_value=0.4,
        observation_value=-0.5,
    )

    assert update.signed_edge < 0.0
    assert update.quality_after < update.quality_before
    assert update.allocation_trust_after == pytest.approx(0.035)
    assert update.allocation_trust_after < update.allocation_trust_before
    assert update.allocation_trust_delta < 0.0


def test_build_cycle_update_clamps_trust_when_quality_and_prior_trust_are_non_positive():
    from alpha_os.policy import build_cycle_update

    update = build_cycle_update(
        quality_before=-0.2,
        allocation_trust_before=0.0,
        prediction_value=0.4,
        observation_value=-0.5,
    )

    assert update.allocation_trust_after == 0.0
    assert update.allocation_trust_delta == 0.0
