from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest


def _evaluation_policy_parts(
    *,
    sizing_method: str = "signal_weighted",
    sizing_engine: str | None = None,
    rebalance_interval_steps: int = 1,
    long_only: bool = False,
    top_k: int | None = None,
    gross_exposure_cap: float | None = None,
    target_vol: float | None = None,
    gross_leverage_cap: float | None = None,
    net_exposure_target: float | None = None,
):
    return {}


def _build_trading_strategy(
    *,
    strategy_id: str,
    label: str,
    subject_set_id: str | None = None,
    target_id: str | None = None,
    sizing_method: str | None = None,
    rebalance: str | None = None,
    long_only: bool | None = None,
    top_k: int | None = None,
    gross_exposure_cap: float | None = None,
    created_at: str = "2026-04-05T00:00:00Z",
):
    from alpha_os.trading_strategy import TradingStrategySpec
    from alpha_os.portfolio_construction_config import (
        PortfolioConstructionSizingSpec,
        PortfolioConstructionSpec,
    )

    return TradingStrategySpec(
        strategy_id=strategy_id,
        label=label,
        subject_set_id=subject_set_id,
        target_id=target_id,
        portfolio_construction=PortfolioConstructionSpec(
            sizing_policy=PortfolioConstructionSizingSpec(
                sizing_method=sizing_method or "equal_weight",
            ),
            direction_mode=(
                "long_only"
                if long_only is True
                else "long_short"
                if long_only is False
                else None
            ),
            gross_exposure_cap=gross_exposure_cap,
        ),
        created_at=created_at,
        rebalance_interval_steps=(
            int(rebalance[len("every_") : -len("_steps")])
            if isinstance(rebalance, str)
            and rebalance.startswith("every_")
            and rebalance.endswith("_steps")
            else 1
        ),
        top_k=top_k,
    )


def test_crypto_regime_momentum_eligibility_requires_trend_confirmation_and_funding_filter():
    from alpha_os.position_rules import (
        crypto_regime_momentum_eligibility_series_by_subject,
    )

    index = pd.date_range("2026-01-01", periods=66, freq="D").strftime("%Y-%m-%d")
    returns = pd.Series(0.01, index=index, dtype=float)
    returns.loc["2026-02-05":"2026-02-10"] = -0.01
    funding_rate = pd.Series(0.001, index=index, dtype=float)
    funding_rate.loc["2026-03-06"] = 0.01

    signals = crypto_regime_momentum_eligibility_series_by_subject(
        subject_return_series_by_subject={"BTC": returns},
        funding_rate_series_by_subject={"BTC": funding_rate},
    )

    signal = signals["BTC"]

    assert signal.loc["2026-01-29"] == pytest.approx(0.0)
    assert signal.loc["2026-01-30"] == pytest.approx(1.0)
    assert signal.loc["2026-02-10"] == pytest.approx(0.0)
    assert signal.loc["2026-03-05"] == pytest.approx(1.0)
    assert signal.loc["2026-03-06"] == pytest.approx(0.0)


def test_crypto_regime_momentum_eligibility_matches_experiment_reference():
    from alpha_os.position_rules import (
        crypto_regime_momentum_eligibility_series_by_subject,
    )

    fixture_dir = Path(__file__).parent / "fixtures" / "crypto_regime_momentum"
    returns_by_subject: dict[str, pd.Series] = {}
    funding_by_subject: dict[str, pd.Series] = {}
    expected_by_subject: dict[str, pd.Series] = {}
    for subject_id in ("BTC", "ETH"):
        frame = pd.read_csv(
            fixture_dir / f"{subject_id}.csv",
            parse_dates=["timestamp"],
        ).sort_values("timestamp")
        frame = frame.set_index("timestamp")
        frame.index = frame.index.tz_convert(None)
        close = frame["close"].astype(float)
        frame["return_7d"] = close / close.shift(7) - 1.0
        frame["return_30d"] = close / close.shift(30) - 1.0
        frame["funding_60d_median"] = frame["funding_rate"].rolling(60).median()
        funding_overheated = (frame["funding_rate"] > 0.0) & (
            frame["funding_rate"] > frame["funding_60d_median"]
        )
        expected_by_subject[subject_id] = (
            ((frame["return_7d"] > 0.0) & (frame["return_30d"] > 0.0) & ~funding_overheated)
            .fillna(False)
            .astype(float)
        )
        returns_by_subject[subject_id] = close.pct_change().dropna()
        funding_by_subject[subject_id] = frame["funding_rate"].astype(float)

    actual_by_subject = crypto_regime_momentum_eligibility_series_by_subject(
        subject_return_series_by_subject=returns_by_subject,
        funding_rate_series_by_subject=funding_by_subject,
    )

    for subject_id, actual in actual_by_subject.items():
        expected = expected_by_subject[subject_id].reindex(actual.index)
        pd.testing.assert_series_equal(
            actual,
            expected,
            check_names=False,
        )


def test_crypto_regime_momentum_eligibility_requires_funding_rate():
    from alpha_os.position_rules import (
        crypto_regime_momentum_eligibility_series_by_subject,
    )

    with pytest.raises(
        ValueError,
        match="crypto regime momentum requires funding_rate series: BTC",
    ):
        crypto_regime_momentum_eligibility_series_by_subject(
            subject_return_series_by_subject={
                "BTC": pd.Series(
                    {"2026-01-01": 0.01},
                    dtype=float,
                )
            },
            funding_rate_series_by_subject={},
        )


def test_direct_strategy_backtest_accepts_position_signal_series(
    monkeypatch,
):
    import alpha_os.strategy_backtest as strategy_backtest
    from alpha_os.evaluation_cost_config import (
        TradingEnvironment,
    )
    from alpha_os.evaluation_spec import EvaluationDateRange
    from alpha_os.portfolio_decision import (
        ObservationSpec,
        SubjectObservationBinding,
        SubjectSet,
    )

    strategy = _build_trading_strategy(
        strategy_id="strategy:crypto_regime_momentum",
        label="Crypto regime momentum",
        subject_set_id="crypto",
        target_id="residual_return_1d",
        long_only=True,
    )
    subject_set = SubjectSet(
        subject_set_id="crypto",
        observation_specs=(
            ObservationSpec(
                observation_spec_id="btc_daily",
                observable_id="daily_close",
            ),
        ),
        bindings=(
            SubjectObservationBinding(
                subject_id="BTC",
                asset="BTC",
                observation_spec_id="btc_daily",
            ),
        ),
    )
    index = pd.date_range("2026-01-01", periods=61, freq="D").strftime("%Y-%m-%d")
    frame_index = pd.date_range("2025-12-31", periods=62, freq="D").strftime("%Y-%m-%d")
    returns = pd.Series(0.01, index=index, dtype=float)
    funding_rate = pd.Series(0.001, index=index, dtype=float)
    close = pd.Series(index=frame_index, dtype=float)
    close.iloc[0] = 100.0
    close.iloc[1:] = (close.iloc[0] * (1.0 + returns).cumprod()).to_numpy()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        strategy_backtest,
        "load_observation_frame",
        lambda *_, **__: pd.DataFrame(
            {
                "timestamp": list(frame_index),
                "close": close.tolist(),
                "funding_rate": [0.001, *funding_rate.tolist()],
            }
        ),
    )

    def capture_metric_group_results(**kwargs):
        captured.update(kwargs)
        return ((), ())

    monkeypatch.setattr(
        strategy_backtest,
        "build_direct_strategy_evaluation_metric_group_results",
        capture_metric_group_results,
    )

    strategy_backtest.run_strategy_backtest(
        subject_set=subject_set,
        target_id="residual_return_1d",
        evaluation_date_ranges=(
            EvaluationDateRange(
                label="eval",
                start_date="2026-01-01",
                end_date="2026-03-02",
            ),
        ),
        base_url="fixture://",
        portfolio_construction=strategy.portfolio_construction,
        trading_environment=TradingEnvironment(),
        top_k=strategy.top_k,
        position_signal_series_by_subject={
            "BTC": pd.Series(
                {
                    "2026-01-29": 0.0,
                    "2026-01-30": 1.0,
                },
                dtype=float,
            )
        },
    )

    signal_series_by_subject = captured["signal_series_by_subject"]
    assert signal_series_by_subject["BTC"].loc["2026-01-29"] == pytest.approx(0.0)
    assert signal_series_by_subject["BTC"].loc["2026-01-30"] == pytest.approx(1.0)
    assert captured["funding_cost_bps_series_by_subject"]["BTC"].iloc[0] == (pytest.approx(10.0))
