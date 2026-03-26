from __future__ import annotations

from .sleeve_scope import with_scope_asset
from .store import HypothesisKind, HypothesisRecord, HypothesisStatus

INITIAL_STAKE = 1.0


def bootstrap_hypotheses() -> list[HypothesisRecord]:
    return _technical_hypotheses() + _ml_hypotheses() + _serious_hypotheses()


def _technical_hypotheses() -> list[HypothesisRecord]:
    items = [
        (
            "technical_rsi_14_reversion",
            "RSI 14 Reversion",
            {
                "indicator": "rsi_reversion",
                "params": {"window": 14, "threshold": 30},
                "inputs": ["core_universe_1000"],
            },
            {"oos_sharpe": 0.45, "oos_log_growth": 0.08},
        ),
        (
            "technical_zscore_60_mean_reversion",
            "Z-Score 60 Mean Reversion",
            {
                "indicator": "zscore_reversion",
                "params": {"window": 60},
                "inputs": ["core_universe_1000"],
            },
            {"oos_sharpe": 0.40, "oos_log_growth": 0.07},
        ),
        (
            "technical_roc_20_momentum",
            "ROC 20 Momentum",
            {
                "indicator": "roc_momentum",
                "params": {"window": 20},
                "inputs": ["core_universe_1000"],
            },
            {"oos_sharpe": 0.55, "oos_log_growth": 0.10},
        ),
        (
            "technical_macd_trend",
            "MACD Trend",
            {
                "indicator": "macd_trend",
                "params": {"fast": 12, "slow": 26, "signal": 9},
                "inputs": ["core_universe_1000"],
            },
            {"oos_sharpe": 0.60, "oos_log_growth": 0.11},
        ),
        (
            "technical_bollinger_reversion",
            "Bollinger Reversion",
            {
                "indicator": "bollinger_reversion",
                "params": {"window": 20, "std": 2.0},
                "inputs": ["core_universe_1000"],
            },
            {"oos_sharpe": 0.42, "oos_log_growth": 0.07},
        ),
        (
            "technical_breakout_60",
            "Breakout 60",
            {
                "indicator": "breakout",
                "params": {"window": 60},
                "inputs": ["core_universe_1000"],
            },
            {"oos_sharpe": 0.58, "oos_log_growth": 0.10},
        ),
        (
            "technical_low_volatility",
            "Low Volatility",
            {
                "indicator": "low_volatility",
                "params": {"window": 20},
                "inputs": ["core_universe_1000"],
            },
            {"oos_sharpe": 0.30, "oos_log_growth": 0.05},
        ),
        (
            "technical_moving_average_cross",
            "Moving Average Cross",
            {
                "indicator": "moving_average_cross",
                "params": {"fast": 20, "slow": 60},
                "inputs": ["core_universe_1000"],
            },
            {"oos_sharpe": 0.52, "oos_log_growth": 0.09},
        ),
        # Intentionally omitted until the runtime has a real volume-backed
        # implementation for this concept.
    ]
    return [
        HypothesisRecord(
            hypothesis_id=hypothesis_id,
            kind=HypothesisKind.TECHNICAL,
            name=name,
            definition=definition,
            status=HypothesisStatus.ACTIVE,
            stake=INITIAL_STAKE,
            source="bootstrap_technical",
            scope=with_scope_asset(None, "BTC"),
            metadata={
                "seed_family": "technical",
                "prior_quality_source": "bootstrap_seed",
                **prior_quality,
            },
        )
        for hypothesis_id, name, definition, prior_quality in items
    ]


def _ml_hypotheses() -> list[HypothesisRecord]:
    items = [
        (
            "ml_linear_residual_v1",
            "Linear Residual V1",
            {
                "model_type": "linear",
                "model_ref": "models/ml_linear_residual_v1.json",
                "features": [
                    "sp500",
                    "nasdaq",
                    "dxy",
                    "fear_greed",
                    "gold",
                    "vix_close",
                ],
            },
            {"oos_sharpe": 0.70, "oos_log_growth": 0.13},
        ),
        (
            "ml_tree_residual_v1",
            "Tree Residual V1",
            {
                "model_type": "tree_ensemble",
                "model_ref": "models/ml_tree_residual_v1.json",
                "features": [
                    "sp500",
                    "nasdaq",
                    "russell2000",
                    "dxy",
                    "gold",
                    "fear_greed",
                    "vix_close",
                ],
            },
            {"oos_sharpe": 0.75, "oos_log_growth": 0.14},
        ),
    ]
    return [
        HypothesisRecord(
            hypothesis_id=hypothesis_id,
            kind=HypothesisKind.ML,
            name=name,
            definition=definition,
            status=HypothesisStatus.ACTIVE,
            stake=INITIAL_STAKE,
            source="bootstrap_ml",
            scope=with_scope_asset(None, "BTC"),
            metadata={
                "seed_family": "ml",
                "prior_quality_source": "bootstrap_seed",
                **prior_quality,
            },
        )
        for hypothesis_id, name, definition, prior_quality in items
    ]


def _serious_hypotheses() -> list[HypothesisRecord]:
    items = [
        (
            "serious_onchain_activity_acceleration_v1",
            "Onchain Activity Acceleration V1",
            "(rank_20 delta_1__btc_active_addresses)",
            "onchain",
            {"oos_sharpe": 0.72, "oos_log_growth": 0.13},
        ),
        (
            "serious_onchain_difficulty_regime_v1",
            "Onchain Difficulty Regime V1",
            "(rank_20 (sign btc_difficulty))",
            "onchain",
            {"oos_sharpe": 0.71, "oos_log_growth": 0.12},
        ),
        (
            "serious_onchain_difficulty_momentum_v1",
            "Onchain Difficulty Momentum V1",
            "(rank_5 (roc_5 btc_difficulty))",
            "onchain",
            {"oos_sharpe": 0.68, "oos_log_growth": 0.11},
        ),
        (
            "serious_derivatives_open_interest_trend_v1",
            "Derivatives Open Interest Trend V1",
            "(mean_20 (rank_10 oi_btc_1h))",
            "derivatives",
            {"oos_sharpe": 0.72, "oos_log_growth": 0.13},
        ),
        (
            "serious_derivatives_funding_crowding_v1",
            "Derivatives Funding Crowding V1",
            "(sub funding_rate_btc (mean_5 funding_rate_btc))",
            "derivatives",
            {"oos_sharpe": 0.66, "oos_log_growth": 0.11},
        ),
        (
            "serious_macro_sentiment_acceleration_v1",
            "Macro Sentiment Acceleration V1",
            "(rank_20 delta_1__fear_greed)",
            "macro",
            {"oos_sharpe": 0.64, "oos_log_growth": 0.10},
        ),
        (
            "serious_macro_dollar_pressure_v1",
            "Macro Dollar Pressure V1",
            "(rank_20 roc_5__dxy)",
            "macro",
            {"oos_sharpe": 0.63, "oos_log_growth": 0.10},
        ),
        (
            "serious_price_regime_shift_v1",
            "Price Regime Shift V1",
            "(rank_20 zscore_20__btc_ohlcv)",
            "price",
            {"oos_sharpe": 0.61, "oos_log_growth": 0.09},
        ),
        (
            "serious_price_short_term_impulse_v1",
            "Price Short-Term Impulse V1",
            "(rank_20 delta_1__btc_ohlcv)",
            "price",
            {"oos_sharpe": 0.60, "oos_log_growth": 0.09},
        ),
    ]
    return [
        HypothesisRecord(
            hypothesis_id=hypothesis_id,
            kind=HypothesisKind.DSL,
            name=name,
            definition={"expression": expression},
            status=HypothesisStatus.ACTIVE,
            stake=0.0,
            source="bootstrap_serious",
            scope=with_scope_asset(None, "BTC"),
            metadata={
                "seed_family": "serious",
                "serious_program": "multi_family_v2",
                "serious_family": serious_family,
                "prior_quality_source": "bootstrap_seed",
                **prior_quality,
            },
        )
        for hypothesis_id, name, expression, serious_family, prior_quality in items
    ]
