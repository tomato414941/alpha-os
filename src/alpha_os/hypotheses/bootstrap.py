from __future__ import annotations

from .store import HypothesisKind, HypothesisRecord, HypothesisStatus

INITIAL_STAKE = 1.0


def bootstrap_hypotheses() -> list[HypothesisRecord]:
    return _technical_hypotheses() + _ml_hypotheses()


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
            metadata={
                "seed_family": "ml",
                "prior_quality_source": "bootstrap_seed",
                **prior_quality,
            },
        )
        for hypothesis_id, name, definition, prior_quality in items
    ]
