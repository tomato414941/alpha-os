import pytest


def test_portfolio_state_exposure_properties():
    from alpha_os.portfolio_decision import PortfolioPositionState, PortfolioState

    state = PortfolioState(
        portfolio_id="paper_core",
        as_of="2026-03-29T00:00:00+00:00",
        positions=(
            PortfolioPositionState(subject_id="BTC", weight=0.3),
            PortfolioPositionState(subject_id="ETH", weight=-0.1),
        ),
        capital_base=2.0,
        gross_limit=1.2,
        net_limit=0.6,
        rebalance_step=4,
    )

    assert state.gross_exposure == pytest.approx(0.4)
    assert state.net_exposure == pytest.approx(0.2)
    assert state.weights_by_subject == {"BTC": 0.3, "ETH": -0.1}
    assert state.capital_base == pytest.approx(2.0)
    assert state.gross_limit == pytest.approx(1.2)
    assert state.net_limit == pytest.approx(0.6)
    assert state.rebalance_step == 4
    assert state.holding_period_days == 0
    assert state.recent_turnover == pytest.approx(0.0)
    assert state.current_drawdown == pytest.approx(0.0)


def test_subject_set_exposes_subject_ids_assets_and_signals():
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )

    subject_set = SubjectSet(
        subject_set_id="core_crypto",
        observation_specs=(
            ObservationSpec(
                observation_spec_id="btc_close",
                observable_id="daily_close",
            ),
            ObservationSpec(
                observation_spec_id="eth_close",
                observable_id="daily_close",
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="BTC_spot",
                asset="BTC",
                observation_spec_id="btc_close",
            ),
            SubjectObservationBinding(
                subject_id="ETH_spot",
                asset="ETH",
                observation_spec_id="eth_close",
            ),
        ),
    )

    assert subject_set.subject_set_id == "core_crypto"
    assert subject_set.subject_ids == ("BTC_spot", "ETH_spot")
    assert subject_set.asset_by_subject == {
        "BTC_spot": "BTC",
        "ETH_spot": "ETH",
    }
    assert subject_set.subject_kind_by_subject == {
        "BTC_spot": "asset",
        "ETH_spot": "asset",
    }
    assert subject_set.observation_spec_id_by_subject == {
        "BTC_spot": "btc_close",
        "ETH_spot": "eth_close",
    }


def test_subject_set_exposes_instrument_metadata_by_subject():
    from alpha_os.portfolio_decision import (
        InstrumentSpec,
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )

    subject_set = SubjectSet(
        subject_set_id="macro_futures",
        instruments=(
            InstrumentSpec(
                instrument_id="es_front",
                instrument_type="future",
                asset="ES",
                venue="CME",
                quote_ccy="USD",
                contract_family="ES",
                asset_class="equity_index",
                region="us",
                liquidity_tier="tier1",
                cluster="eq_index_dm",
                roll_rule="volume_switch",
                multiplier=50.0,
            ),
        ),
        observation_specs=(
            ObservationSpec(
                observation_spec_id="es_close",
                observable_id="daily_close",
                provided_observable_ids=(
                    "front_price",
                    "next_price",
                    "basis",
                ),
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="ES_front",
                subject_kind="future",
                asset="ES",
                observation_spec_id="es_close",
                instrument_id="es_front",
            ),
        ),
    )

    assert subject_set.instrument_id_by_subject == {"ES_front": "es_front"}
    instrument = subject_set.instrument_for_subject("ES_front")
    assert instrument is not None
    assert instrument.instrument_type == "future"
    assert instrument.venue == "CME"
    assert subject_set.asset_class_by_subject == {"ES_front": "equity_index"}
    assert subject_set.region_by_subject == {"ES_front": "us"}
    assert subject_set.liquidity_tier_by_subject == {"ES_front": "tier1"}
    assert subject_set.cluster_by_subject == {"ES_front": "eq_index_dm"}
    assert subject_set.subjects_grouped_by_instrument_field("asset_class") == {
        "equity_index": ("ES_front",)
    }
    assert (
        subject_set.observation_spec_for_subject("ES_front").provided_observable_ids
        == ("front_price", "next_price", "basis")
    )


def test_subject_set_supports_multiple_subject_kinds_without_backend_names():
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )

    subject_set = SubjectSet(
        subject_set_id="macro_mix",
        observation_specs=(
            ObservationSpec(
                observation_spec_id="spy_close",
                observable_id="daily_close",
            ),
            ObservationSpec(
                observation_spec_id="vix_close",
                observable_id="daily_close",
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="SPY_spot",
                asset="SPY",
                observation_spec_id="spy_close",
                subject_kind="equity",
            ),
            SubjectObservationBinding(
                subject_id="VIX_index",
                asset="VIX",
                observation_spec_id="vix_close",
                subject_kind="index",
            ),
        ),
    )

    assert subject_set.subject_kind_by_subject == {
        "SPY_spot": "equity",
        "VIX_index": "index",
    }
    assert subject_set.observation_spec_id_by_subject == {
        "SPY_spot": "spy_close",
        "VIX_index": "vix_close",
    }


