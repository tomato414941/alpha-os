from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from alpha_os.trading_strategy import TradingStrategy

from strategies.crypto_market_structure.data import MarketStructureDay


@dataclass(frozen=True)
class MarketStructureDecisionInput:
    history_by_symbol: dict[str, tuple[MarketStructureDay, ...]]
    current_weights: dict[str, float]
    equity: float


@dataclass(frozen=True)
class MarketStructureTargetWeights:
    target_weights: dict[str, float]


@dataclass(frozen=True)
class MarketStructureRankStrategy(
    TradingStrategy[MarketStructureDecisionInput, MarketStructureTargetWeights]
):
    feature_weights: dict[str, float]
    top_n: int = 2

    def decide(
        self,
        strategy_input: MarketStructureDecisionInput,
    ) -> MarketStructureTargetWeights:
        feature_values = _feature_values(strategy_input.history_by_symbol)
        if not feature_values:
            return MarketStructureTargetWeights(target_weights={})
        symbols = sorted(
            set.intersection(*(set(values) for values in feature_values.values()))
        )
        scores = {
            symbol: sum(
                weight * _zscore(feature_values[feature], symbol)
                for feature, weight in self.feature_weights.items()
                if feature in feature_values
            )
            for symbol in symbols
        }
        selected = tuple(
            symbol
            for symbol, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score > 0.0
        )[: self.top_n]
        if not selected:
            return MarketStructureTargetWeights(target_weights={})
        weight = 1.0 / len(selected)
        return MarketStructureTargetWeights(
            target_weights={symbol: weight for symbol in selected}
        )


def _feature_values(
    history_by_symbol: dict[str, tuple[MarketStructureDay, ...]],
) -> dict[str, dict[str, float]]:
    latest_by_symbol = {
        symbol: rows[-1]
        for symbol, rows in history_by_symbol.items()
        if rows and rows[-1].volume > 0.0
    }
    if not latest_by_symbol:
        return {}
    return {
        "funding_rate_sum": {
            symbol: row.funding_rate_sum
            for symbol, row in latest_by_symbol.items()
        },
        "premium_close": {
            symbol: row.premium_close
            for symbol, row in latest_by_symbol.items()
        },
        "taker_buy_imbalance": {
            symbol: (row.taker_buy_volume / row.volume) - 0.5
            for symbol, row in latest_by_symbol.items()
        },
        "volume_ratio_20d": {
            symbol: _volume_ratio(rows)
            for symbol, rows in history_by_symbol.items()
            if rows and rows[-1].volume > 0.0
        },
    }


def _volume_ratio(rows: tuple[MarketStructureDay, ...]) -> float:
    window = rows[-20:]
    average_volume = mean(row.volume for row in window)
    return rows[-1].volume / average_volume if average_volume > 0.0 else 1.0


def _zscore(values_by_symbol: dict[str, float], symbol: str) -> float:
    values = tuple(values_by_symbol.values())
    if symbol not in values_by_symbol or len(values) < 2:
        return 0.0
    average = mean(values)
    variance = mean((value - average) ** 2 for value in values)
    if variance <= 0.0:
        return 0.0
    return (values_by_symbol[symbol] - average) / (variance**0.5)
