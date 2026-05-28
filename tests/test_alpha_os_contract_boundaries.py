from __future__ import annotations

from alpha_os.contract_boundaries import (
    default_portfolio_constraint_boundary,
    default_subject_set_contract_boundary,
    format_active_constraint_stages,
    format_subject_set_contract_groups,
)
from alpha_os.portfolio_construction_config import (
    PortfolioConstructionSpec,
)
from alpha_os.portfolio_decision import SubjectSet


def test_subject_set_contract_boundary_names_field_owners():
    boundary = default_subject_set_contract_boundary()

    assert boundary.group_for_field("instrument.instrument_type") == "instrument"
    assert boundary.group_for_field("observation_spec.observable_id") == "observation_spec"
    assert boundary.group_for_field("binding.instrument_id") == "binding"
    assert boundary.group_for_field("universe_policy.base_currency") == "universe_policy"
    assert boundary.group_for_field("binding.unknown") is None


def test_subject_set_exposes_canonical_contract_boundary():
    subject_set = SubjectSet(subject_set_id="macro_core")

    assert subject_set.contract_boundary == default_subject_set_contract_boundary()
    assert format_subject_set_contract_groups(subject_set.contract_boundary) == (
        "instrument,observation_spec,binding,universe_policy"
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


def test_portfolio_construction_exposes_canonical_constraint_boundary():
    construction = PortfolioConstructionSpec(
        gross_exposure_cap=1.0,
        target_vol=0.12,
        gross_leverage_cap=1.5,
        net_exposure_target=0.3,
    )

    assert construction.constraint_boundary == default_portfolio_constraint_boundary()
