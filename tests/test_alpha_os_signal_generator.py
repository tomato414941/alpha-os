from __future__ import annotations


def test_generate_signal_discovery_builds_diverse_families():
    from alpha_os.signal_generator import (
        SignalDiscoveryGenerationConstraint,
        SignalDiscoveryGenerationSpec,
        generate_signal_discovery,
    )
    from alpha_os.signal_discovery import SignalParameterSpace

    discovery = generate_signal_discovery(
        SignalDiscoveryGenerationSpec(
            signal_discovery_id="generated_search",
            subject_set_id="us_equity_core",
            operator_ids=(
                "trend",
                "volatility_breakout",
                "volume_confirmed_trend",
                "relative_strength",
            ),
            primary_observable_ids=(
                "daily_close",
                "cross_sectional_return_rank_20d",
            ),
            secondary_observable_ids=("dollar_volume_20d",),
            conditioning_observable_ids=("realized_vol_20d",),
            parameter_space=SignalParameterSpace.from_document(
                {"lookback": [20, 40]}
            ),
            target_id="residual_return_3d",
            constraint=SignalDiscoveryGenerationConstraint(
                max_families_per_operator=1,
            ),
        )
    )

    families = {family.family_id: family for family in discovery.families}
    assert len(families) == 4
    assert "trend__daily_close" in families
    assert (
        "volatility_breakout__daily_close__realized_vol_20d"
        in families
    )
    assert (
        "volume_confirmed_trend__daily_close__dollar_volume_20d"
        in families
    )
    assert "relative_strength__cross_sectional_return_rank_20d" in families
    assert families["relative_strength__cross_sectional_return_rank_20d"].family_group == (
        "cross_sectional"
    )


def test_generate_signal_discovery_respects_generation_constraints():
    from alpha_os.signal_generator import (
        SignalDiscoveryGenerationConstraint,
        SignalDiscoveryGenerationSpec,
        generate_signal_discovery,
    )
    from alpha_os.signal_discovery import SignalParameterSpace

    discovery = generate_signal_discovery(
        SignalDiscoveryGenerationSpec(
            signal_discovery_id="generated_search",
            subject_set_id="us_equity_core",
            operator_ids=("trend", "mean_reversion", "relative_strength"),
            primary_observable_ids=("daily_close", "cross_sectional_return_rank_20d"),
            parameter_space=SignalParameterSpace.from_document(
                {"lookback": [20, 40]}
            ),
            constraint=SignalDiscoveryGenerationConstraint(
                max_families_total=2,
                allowed_family_groups=("price", "cross_sectional"),
            ),
        )
    )

    assert len(discovery.families) == 2
    assert all(
        family.family_group in {"price", "cross_sectional"}
        for family in discovery.families
    )


def test_materialize_signal_specs_from_generated_discovery():
    from alpha_os.signal_generator import (
        SignalDiscoveryGenerationSpec,
        generate_signal_discovery,
        materialize_signal_specs,
    )
    from alpha_os.signal_discovery import SignalParameterSpace

    discovery = generate_signal_discovery(
        SignalDiscoveryGenerationSpec(
            signal_discovery_id="generated_search",
            subject_set_id="us_equity_core",
            operator_ids=("trend", "relative_strength"),
            primary_observable_ids=("daily_close", "cross_sectional_return_rank_20d"),
            parameter_space=SignalParameterSpace.from_document(
                {"lookback": [20, 40]}
            ),
            target_id="residual_return_3d",
        )
    )

    specifications = materialize_signal_specs(discovery)
    specification_ids = {definition.signal_id for definition in specifications}
    assert "trend__daily_close__lookback_20" in specification_ids
    assert (
        "relative_strength__cross_sectional_return_rank_20d__lookback_40"
        in specification_ids
    )


def test_materialize_signal_specs_uses_discovery_target_fallback():
    from alpha_os.signal_discovery import (
        SignalDiscoverySpec,
        SignalFamily,
        SignalParameterSpace,
    )
    from alpha_os.signal_generator import materialize_signal_specs

    specifications = materialize_signal_specs(
        SignalDiscoverySpec(
            signal_discovery_id="generated_search",
            subject_set_id="us_equity_core",
            families=(
                SignalFamily(
                    family_id="trend_family",
                    kind="trend",
                    parameter_space=SignalParameterSpace.from_document(
                        {"lookback": [20]}
                    ),
                    required_observable_id="daily_close",
                    target_id=None,
                ),
            ),
            target_id="residual_return_1d",
        )
    )

    assert specifications[0].target_id == "residual_return_1d"
    assert specifications[0].horizon_days == 1


def test_generate_signal_discovery_can_limit_primary_observable_reuse():
    from alpha_os.signal_generator import (
        SignalDiscoveryGenerationConstraint,
        SignalDiscoveryGenerationSpec,
        generate_signal_discovery,
    )
    from alpha_os.signal_discovery import SignalParameterSpace

    discovery = generate_signal_discovery(
        SignalDiscoveryGenerationSpec(
            signal_discovery_id="generated_search",
            subject_set_id="us_equity_core",
            operator_ids=(
                "trend",
                "mean_reversion",
                "volatility_breakout",
            ),
            primary_observable_ids=("daily_close",),
            conditioning_observable_ids=("realized_vol_20d",),
            parameter_space=SignalParameterSpace.from_document(
                {"lookback": [20, 40]}
            ),
            constraint=SignalDiscoveryGenerationConstraint(
                max_families_per_primary_observable=2,
            ),
        )
    )

    assert len(discovery.families) == 2
    assert {
        family.required_observable_id for family in discovery.families
    } == {"daily_close"}


def test_generate_signal_discovery_can_use_coarser_novelty_budget():
    from alpha_os.signal_generator import (
        SignalDiscoveryGenerationConstraint,
        SignalDiscoveryGenerationSpec,
        generate_signal_discovery,
    )
    from alpha_os.signal_discovery import SignalParameterSpace

    discovery = generate_signal_discovery(
        SignalDiscoveryGenerationSpec(
            signal_discovery_id="generated_search",
            subject_set_id="us_equity_core",
            operator_ids=("low_vol_trend", "post_shock_reversion"),
            primary_observable_ids=("daily_close",),
            conditioning_observable_ids=("realized_vol_20d", "market_vol_regime_20d"),
            parameter_space=SignalParameterSpace.from_document(
                {"lookback": [20]}
            ),
            constraint=SignalDiscoveryGenerationConstraint(
                novelty_key="family_group_primary",
            ),
        )
    )

    assert len(discovery.families) == 1
    assert discovery.families[0].family_group == "regime_conditioned"


def test_generate_signal_discovery_can_use_operator_primary_novelty_budget():
    from alpha_os.signal_generator import (
        SignalDiscoveryGenerationConstraint,
        SignalDiscoveryGenerationSpec,
        generate_signal_discovery,
    )
    from alpha_os.signal_discovery import SignalParameterSpace

    discovery = generate_signal_discovery(
        SignalDiscoveryGenerationSpec(
            signal_discovery_id="generated_search",
            subject_set_id="us_equity_core",
            operator_ids=("trend", "mean_reversion"),
            primary_observable_ids=("daily_close",),
            parameter_space=SignalParameterSpace.from_document(
                {"lookback": [20]}
            ),
            constraint=SignalDiscoveryGenerationConstraint(
                novelty_key="operator_primary",
                max_families_per_primary_observable=2,
            ),
        )
    )

    assert {family.kind for family in discovery.families} == {
        "momentum",
        "reversal",
    }


def test_generate_signal_discovery_supports_instrument_aware_operators():
    from alpha_os.signal_generator import (
        SignalDiscoveryGenerationSpec,
        generate_signal_discovery,
    )
    from alpha_os.signal_discovery import SignalParameterSpace

    discovery = generate_signal_discovery(
        SignalDiscoveryGenerationSpec(
            signal_discovery_id="macro_search",
            subject_set_id="macro_futures",
            operator_ids=(
                "time_series_trend",
                "term_structure_carry",
                "basis_carry",
                "value_anchor",
            ),
            primary_observable_ids=(
                "daily_close",
                "term_structure_slope",
                "basis",
                "valuation_ratio",
            ),
            parameter_space=SignalParameterSpace.from_document({"lookback": [20]}),
        )
    )

    families = {family.family_id: family for family in discovery.families}
    assert "time_series_trend__daily_close" in families
    assert "term_structure_carry__term_structure_slope" in families
    assert "basis_carry__basis" in families
    assert "value_anchor__valuation_ratio" in families
    assert families["term_structure_carry__term_structure_slope"].family_group == "carry"
