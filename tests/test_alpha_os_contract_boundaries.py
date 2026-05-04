from __future__ import annotations

import pytest

from alpha_os.contract_boundaries import (
    default_portfolio_constraint_boundary,
    default_subject_set_contract_boundary,
    format_active_constraint_stages,
    format_subject_set_contract_groups,
)
from alpha_os.portfolio_construction_config import (
    PortfolioConstructionSpec,
    PortfolioRiskBudgetSpec,
)
from alpha_os.portfolio_decision import SubjectSet
from alpha_os.trading_strategy import RiskPolicySpec


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
            "long_only": True,
            "gross_exposure_cap": 1.0,
            "target_vol": 0.12,
            "gross_leverage_cap": 1.5,
            "net_exposure_target": 0.3,
            "asset_class_weight_caps": {},
            "cluster_weight_caps": {"crypto": 0.5},
        },
    ) == (
        "sizing_time:target_vol;"
        "post_sizing_normalization:long_only,gross_exposure_cap,"
        "gross_leverage_cap,net_exposure_target,cluster_weight_caps"
    )


def test_risk_policy_exposes_canonical_constraint_boundary():
    risk_policy = RiskPolicySpec(
        long_only=True,
        gross_exposure_cap=1.0,
        target_vol=0.12,
        gross_leverage_cap=1.5,
        net_exposure_target=0.3,
    )

    assert risk_policy.constraint_boundary == default_portfolio_constraint_boundary()


def test_portfolio_construction_exposes_canonical_constraint_boundary():
    construction = PortfolioConstructionSpec(
        gross_exposure_cap=1.0,
        target_vol=0.12,
        gross_leverage_cap=1.5,
        net_exposure_target=0.3,
    )

    assert construction.constraint_boundary == default_portfolio_constraint_boundary()


def test_portfolio_construction_defaults_to_rank_tilt_overlay():
    construction = PortfolioConstructionSpec()
    document = construction.to_document()
    restored = PortfolioConstructionSpec.from_document(document)

    assert "top_k_mode" not in document
    assert "top_k_tilt_fraction" not in document
    assert construction.active_overlay is not None
    assert construction.active_overlay.kind == "rank_tilt"
    assert construction.active_overlay.active_weight_budget == pytest.approx(0.30)
    assert restored.active_overlay is not None
    assert restored.active_overlay.kind == "rank_tilt"
    assert restored.active_overlay.active_weight_budget == pytest.approx(0.30)


def test_portfolio_construction_roundtrips_active_overlay():
    from alpha_os.portfolio_overlay import ActiveOverlaySpec

    construction = PortfolioConstructionSpec(
        active_overlay=ActiveOverlaySpec(active_weight_budget=0.2)
    )
    restored = PortfolioConstructionSpec.from_document(construction.to_document())

    assert restored.active_overlay is not None
    assert restored.active_overlay.kind == "rank_tilt"
    assert restored.active_overlay.active_weight_budget == pytest.approx(0.2)


def test_portfolio_construction_roundtrips_portfolio_intent():
    from alpha_os.portfolio_construction_config import PortfolioIntentSpec

    construction = PortfolioConstructionSpec(
        portfolio_intent=PortfolioIntentSpec(
            effective_n_floor=8.0,
            top_gross_share_cap_n=3,
            top_gross_share_cap=0.55,
        )
    )
    document = construction.to_document()
    restored = PortfolioConstructionSpec.from_document(document)

    assert document["portfolio_intent"] == {
        "effective_n_floor": 8.0,
        "top_gross_share_cap_n": 3,
        "top_gross_share_cap": 0.55,
    }
    assert restored.portfolio_intent.effective_n_floor == pytest.approx(8.0)
    assert restored.portfolio_intent.top_gross_share_cap_n == 3
    assert restored.portfolio_intent.top_gross_share_cap == pytest.approx(0.55)


def test_portfolio_construction_roundtrips_risk_budget():
    construction = PortfolioConstructionSpec(
        risk_budget=PortfolioRiskBudgetSpec(
            risk_normalization_mode="gross",
            target_gross_exposure=0.5,
            allow_releverage=True,
        )
    )
    document = construction.to_document()
    restored = PortfolioConstructionSpec.from_document(document)

    assert document["risk_budget"] == {
        "risk_normalization_mode": "gross",
        "target_gross_exposure": 0.5,
        "allow_releverage": True,
    }
    assert restored.risk_budget.risk_normalization_mode == "gross"
    assert restored.risk_budget.target_gross_exposure == pytest.approx(0.5)
    assert restored.risk_budget.allow_releverage is True


def test_portfolio_risk_budget_rejects_unknown_normalization_mode():
    with pytest.raises(ValueError, match="risk_normalization_mode"):
        PortfolioRiskBudgetSpec(risk_normalization_mode="unsupported")


def test_portfolio_risk_budget_parses_string_boolean_without_truthy_leak():
    restored = PortfolioRiskBudgetSpec.from_document(
        {
            "risk_normalization_mode": "gross",
            "target_gross_exposure": 0.5,
            "allow_releverage": "false",
        }
    )

    assert restored.allow_releverage is False
