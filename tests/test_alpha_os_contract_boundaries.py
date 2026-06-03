from __future__ import annotations

from alpha_os.contract_boundaries import (
    default_portfolio_constraint_boundary,
    format_active_constraint_stages,
)


def test_portfolio_constraint_boundary_marks_enforcement_stages():
    boundary = default_portfolio_constraint_boundary()

    assert boundary.stage_for_field("target_vol") == "sizing_time"
    assert boundary.stage_for_field("gross_exposure_cap") == "post_sizing_normalization"
    assert boundary.stage_for_field("gross_leverage_cap") == "post_sizing_normalization"
    assert boundary.stage_for_field("net_exposure_target") == "post_sizing_normalization"
    assert boundary.stage_for_field("cluster_weight_caps") == "post_sizing_normalization"
    assert boundary.stage_for_field("unknown") is None


def test_portfolio_constraint_boundary_formats_active_stages():
    boundary = default_portfolio_constraint_boundary()

    assert format_active_constraint_stages(
        boundary,
        field_values={
            "direction_mode": "long_only",
            "gross_exposure_cap": 1.0,
            "target_vol": 0.12,
            "gross_leverage_cap": 1.5,
            "net_exposure_target": 0.3,
            "asset_class_weight_caps": {},
            "cluster_weight_caps": {"crypto": 0.5},
        },
    ) == (
        "sizing_time:target_vol;"
        "post_sizing_normalization:direction_mode,gross_exposure_cap,"
        "gross_leverage_cap,net_exposure_target,cluster_weight_caps"
    )
