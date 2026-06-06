from __future__ import annotations

from math import sqrt

from strategies.daily_close.backtest import DailyCloseDecisionInput, TargetWeights


class CrossAssetTopMomentumStrategy:
    def __init__(
        self,
        *,
        momentum_lookback_days: int = 126,
        trend_lookback_days: int = 252,
    ) -> None:
        self._momentum_lookback_days = momentum_lookback_days
        self._trend_lookback_days = trend_lookback_days

    def decide(self, strategy_input: DailyCloseDecisionInput) -> TargetWeights:
        symbol = _best_positive_momentum_symbol(
            strategy_input.closes_by_symbol,
            momentum_lookback_days=self._momentum_lookback_days,
            trend_lookback_days=self._trend_lookback_days,
        )
        return TargetWeights(target_weights={symbol: 1.0} if symbol else {})


class CrossAssetVolAdjustedMomentumStrategy:
    def __init__(
        self,
        *,
        momentum_lookback_days: int = 126,
        trend_lookback_days: int = 252,
        volatility_lookback_days: int = 63,
    ) -> None:
        self._momentum_lookback_days = momentum_lookback_days
        self._trend_lookback_days = trend_lookback_days
        self._volatility_lookback_days = volatility_lookback_days

    def decide(self, strategy_input: DailyCloseDecisionInput) -> TargetWeights:
        required_closes = max(
            self._momentum_lookback_days,
            self._trend_lookback_days,
            self._volatility_lookback_days,
        ) + 1
        best_symbol = ""
        best_score = 0.0
        for symbol, closes in strategy_input.closes_by_symbol.items():
            if len(closes) < required_closes:
                continue
            momentum = _lookback_return(closes, self._momentum_lookback_days)
            trend = _lookback_return(closes, self._trend_lookback_days)
            volatility = _realized_volatility(closes, self._volatility_lookback_days)
            if momentum <= 0.0 or trend <= 0.0 or volatility <= 0.0:
                continue
            score = momentum / volatility
            if score > best_score:
                best_symbol = symbol
                best_score = score
        return TargetWeights(target_weights={best_symbol: 1.0} if best_symbol else {})


class CrossAssetRiskOnOffStrategy:
    def __init__(
        self,
        *,
        risk_symbols: tuple[str, ...] = ("QQQ", "BTCUSDT", "ETHUSDT"),
        defensive_symbols: tuple[str, ...] = ("GLD", "TLT"),
        regime_symbol: str = "SPY",
        lookback_days: int = 126,
    ) -> None:
        self._risk_symbols = risk_symbols
        self._defensive_symbols = defensive_symbols
        self._regime_symbol = regime_symbol
        self._lookback_days = lookback_days

    def decide(self, strategy_input: DailyCloseDecisionInput) -> TargetWeights:
        regime_return = _lookback_return(
            strategy_input.closes_by_symbol.get(self._regime_symbol, ()),
            self._lookback_days,
        )
        candidate_symbols = (
            self._risk_symbols
            if regime_return > 0.0
            else self._defensive_symbols
        )
        symbol = _best_positive_momentum_symbol(
            {
                candidate: strategy_input.closes_by_symbol.get(candidate, ())
                for candidate in candidate_symbols
            },
            momentum_lookback_days=self._lookback_days,
            trend_lookback_days=self._lookback_days,
        )
        return TargetWeights(target_weights={symbol: 1.0} if symbol else {})


def _best_positive_momentum_symbol(
    closes_by_symbol: dict[str, tuple[float, ...]],
    *,
    momentum_lookback_days: int,
    trend_lookback_days: int,
) -> str:
    required_closes = max(momentum_lookback_days, trend_lookback_days) + 1
    best_symbol = ""
    best_momentum = 0.0
    for symbol, closes in closes_by_symbol.items():
        if len(closes) < required_closes:
            continue
        momentum = _lookback_return(closes, momentum_lookback_days)
        trend = _lookback_return(closes, trend_lookback_days)
        if trend > 0.0 and momentum > best_momentum:
            best_symbol = symbol
            best_momentum = momentum
    return best_symbol


def _lookback_return(closes: tuple[float, ...], lookback_days: int) -> float:
    if len(closes) < lookback_days + 1:
        return 0.0
    previous_close = closes[-lookback_days - 1]
    current_close = closes[-1]
    return (current_close / previous_close) - 1.0 if previous_close > 0.0 else 0.0


def _realized_volatility(closes: tuple[float, ...], lookback_days: int) -> float:
    if len(closes) < lookback_days + 1:
        return 0.0
    window = closes[-lookback_days - 1 :]
    returns = tuple(
        (current / previous) - 1.0
        for previous, current in zip(window[:-1], window[1:], strict=True)
        if previous > 0.0
    )
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return sqrt(variance)
