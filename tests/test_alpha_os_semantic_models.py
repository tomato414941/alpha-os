from __future__ import annotations

import pandas as pd
import pytest


def test_observable_definitions_include_non_price_semantics():
    from alpha_os.observables import find_observable_definition, list_observable_definitions

    observable_ids = {definition.observable_id for definition in list_observable_definitions()}
    assert {
        "daily_close",
        "daily_return",
        "realized_vol_20d",
        "daily_volume",
        "dollar_volume_20d",
        "cross_sectional_return_rank_20d",
        "market_vol_regime_20d",
        "front_price",
        "next_price",
        "term_structure_slope",
        "funding_rate",
        "open_interest",
        "basis",
        "borrow_fee",
        "valuation_ratio",
        "earnings_revision",
    }.issubset(observable_ids)

    realized_vol = find_observable_definition("realized_vol_20d")
    assert realized_vol is not None
    assert realized_vol.input_observable_ids == ("daily_return",)
    assert "equity" in realized_vol.applicable_subject_kinds


def test_observable_definition_round_trips_semantic_metadata():
    from alpha_os.observables import ObservableDefinition

    definition = ObservableDefinition(
        observable_id="peer_spread_20d",
        family="cross_sectional",
        value_kind="real_value",
        default_resolution="1d",
        params={"lookback": 20},
        description="Peer spread over twenty days.",
        input_observable_ids=("daily_return", "cross_sectional_return_rank_20d"),
        applicable_subject_kinds=("equity", "etf"),
    )

    restored = ObservableDefinition.from_document(definition.to_document())
    assert restored == definition


def test_signal_family_round_trips_greenfield_metadata():
    from alpha_os.signal_discovery import (
        SignalFamily,
        SignalParameterSpace,
    )

    family = SignalFamily(
        family_id="trend_with_volume_confirmation_family",
        kind="momentum",
        parameter_space=SignalParameterSpace.from_document(
            {
                "lookback": [20, 40],
                "confirmation_window": [5],
            }
        ),
        required_observable_id="daily_close",
        family_group="interaction",
        secondary_observable_ids=("dollar_volume_20d",),
        conditioning_observable_ids=("market_vol_regime_20d",),
        applicable_subject_kinds=("equity", "etf"),
        thesis="Trend is stronger when liquidity expands in calm regimes.",
    )

    restored = SignalFamily.from_document(family.to_document())
    assert restored == family


def test_prepare_feature_plane_from_frame_collects_instrument_observables():
    from alpha_os.feature_plane_builder import prepare_feature_plane_from_frame

    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-24T00:00:00+00:00",
                "close": 100.0,
                "front_price": 100.0,
                "next_price": 101.0,
                "funding_rate": 0.0001,
                "open_interest": 1000.0,
                "basis": 0.01,
            },
            {
                "timestamp": "2026-03-25T00:00:00+00:00",
                "close": 102.0,
                "front_price": 102.0,
                "next_price": 103.53,
                "funding_rate": 0.0002,
                "open_interest": 1010.0,
                "basis": 0.015,
                "roll_cost_bps": 3.5,
            },
        ]
    )

    plane = prepare_feature_plane_from_frame(frame=frame)

    assert plane.observable_series(observable_id="funding_rate").loc["2026-03-25"] == 0.0002
    assert plane.observable_series(observable_id="open_interest").loc["2026-03-24"] == 1000.0
    assert plane.observable_series(observable_id="basis").loc["2026-03-25"] == 0.015
    assert plane.observable_series(observable_id="roll_cost_bps").loc["2026-03-25"] == 3.5
    assert plane.observable_series(observable_id="term_structure_slope").loc["2026-03-25"] == pytest.approx(0.015)


def test_feature_plane_supports_new_signal_kinds():
    from alpha_os.feature_plane_builder import prepare_feature_plane_from_frame

    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-20T00:00:00+00:00",
                "close": 100.0,
                "front_price": 100.0,
                "next_price": 101.0,
                "funding_rate": 0.0001,
                "basis": 0.01,
                "valuation_ratio": 1.2,
            },
            {
                "timestamp": "2026-03-21T00:00:00+00:00",
                "close": 101.0,
                "front_price": 101.0,
                "next_price": 102.515,
                "funding_rate": 0.0002,
                "basis": 0.012,
                "valuation_ratio": 1.3,
            },
            {
                "timestamp": "2026-03-22T00:00:00+00:00",
                "close": 103.0,
                "front_price": 103.0,
                "next_price": 104.03,
                "funding_rate": 0.0003,
                "basis": 0.013,
                "valuation_ratio": 1.4,
            },
        ]
    )

    plane = prepare_feature_plane_from_frame(frame=frame)

    assert plane.signal_series(kind="time_series_trend", lookback=2).loc["2026-03-22"] > 0.0
    assert plane.signal_series(kind="term_structure_carry", lookback=2).loc["2026-03-22"] == pytest.approx(0.0125)
    assert plane.signal_series(kind="funding_carry", lookback=2).loc["2026-03-22"] == pytest.approx(0.00025)
    assert plane.signal_series(kind="basis_carry", lookback=2).loc["2026-03-22"] == pytest.approx(0.0125)
    assert plane.signal_series(kind="value_anchor", lookback=2).loc["2026-03-22"] < 0.0
