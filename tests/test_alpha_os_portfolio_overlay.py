from __future__ import annotations

import pytest


def _target(subject_id: str, weight: float):
    from alpha_os.portfolio_decision import PortfolioTarget

    return PortfolioTarget(
        subject_id=subject_id,
        target_weight=weight,
        position_delta=0.0,
    )


def _weights(targets):
    return {item.subject_id: item.target_weight for item in targets}


def test_rank_tilt_overlay_overweights_high_rank_long_only_without_zeroing():
    from alpha_os.portfolio_overlay import ActiveOverlaySpec, apply_active_overlay

    adjusted = apply_active_overlay(
        [_target("A", 0.5), _target("B", 0.3), _target("C", 0.2)],
        spec=ActiveOverlaySpec(active_weight_budget=0.3),
        direction_mode="long_only",
    )

    weights = _weights(adjusted)
    assert weights["A"] > 0.5
    assert weights["B"] == pytest.approx(0.3)
    assert 0.0 < weights["C"] < 0.2


def test_rank_tilt_overlay_tilts_short_only_without_flipping_long():
    from alpha_os.portfolio_overlay import ActiveOverlaySpec, apply_active_overlay

    adjusted = apply_active_overlay(
        [_target("A", -0.5), _target("B", -0.3), _target("C", -0.2)],
        spec=ActiveOverlaySpec(active_weight_budget=0.3),
        direction_mode="short_only",
    )

    weights = _weights(adjusted)
    assert weights["A"] < -0.5
    assert weights["B"] == pytest.approx(-0.3)
    assert -0.2 < weights["C"] < 0.0


def test_rank_tilt_overlay_uses_absolute_conviction_for_long_short():
    from alpha_os.portfolio_overlay import ActiveOverlaySpec, apply_active_overlay

    adjusted = apply_active_overlay(
        [_target("SMALL_LONG", 0.2), _target("BIG_SHORT", -0.8)],
        spec=ActiveOverlaySpec(active_weight_budget=0.3),
        direction_mode="long_short",
    )

    weights = _weights(adjusted)
    assert 0.0 < weights["SMALL_LONG"] < 0.2
    assert weights["BIG_SHORT"] < -0.8


def test_rank_tilt_overlay_noops_when_no_cross_sectional_rank_exists():
    from alpha_os.portfolio_overlay import ActiveOverlaySpec, apply_active_overlay

    cases = (
        ([_target("A", 0.5)], "long_only"),
        ([_target("A", 0.5), _target("B", 0.5)], "long_only"),
        ([_target("A", 0.0), _target("B", 0.0)], "long_short"),
    )

    for targets, direction_mode in cases:
        adjusted = apply_active_overlay(
            targets,
            spec=ActiveOverlaySpec(active_weight_budget=0.3),
            direction_mode=direction_mode,
        )
        assert _weights(adjusted) == pytest.approx(_weights(targets))


def test_rank_tilt_overlay_noops_when_disabled():
    from alpha_os.portfolio_overlay import ActiveOverlaySpec, apply_active_overlay

    targets = [_target("A", 0.5), _target("B", 0.3), _target("C", 0.2)]

    assert _weights(
        apply_active_overlay(
            targets,
            spec=None,
            direction_mode="long_only",
        )
    ) == pytest.approx(_weights(targets))
    assert _weights(
        apply_active_overlay(
            targets,
            spec=ActiveOverlaySpec(active_weight_budget=0.0),
            direction_mode="long_only",
        )
    ) == pytest.approx(_weights(targets))
