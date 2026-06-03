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
        Subject,
        SubjectSet,
    )

    subject_set = SubjectSet(
        subject_set_id="core_crypto",
        subjects=(
            Subject(
                subject_id="BTC_spot",
                asset="BTC",
            ),
            Subject(
                subject_id="ETH_spot",
                asset="ETH",
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


def test_subject_set_supports_multiple_subject_kinds_without_backend_names():
    from alpha_os.portfolio_decision import (
        Subject,
        SubjectSet,
    )

    subject_set = SubjectSet(
        subject_set_id="macro_mix",
        subjects=(
            Subject(
                subject_id="SPY_spot",
                asset="SPY",
                subject_kind="equity",
            ),
            Subject(
                subject_id="VIX_index",
                asset="VIX",
                subject_kind="index",
            ),
        ),
    )

    assert subject_set.subject_kind_by_subject == {
        "SPY_spot": "equity",
        "VIX_index": "index",
    }
