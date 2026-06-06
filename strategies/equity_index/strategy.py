from __future__ import annotations

from strategies.daily_close.backtest import DailyCloseDecisionInput, TargetWeights


class PositiveTrendTopMomentumStrategy:
    def __init__(
        self,
        *,
        momentum_lookback_days: int = 63,
        trend_lookback_days: int = 126,
    ) -> None:
        self._momentum_lookback_days = momentum_lookback_days
        self._trend_lookback_days = trend_lookback_days

    def decide(self, strategy_input: DailyCloseDecisionInput) -> TargetWeights:
        required_closes = max(
            self._momentum_lookback_days,
            self._trend_lookback_days,
        ) + 1
        best_symbol = ""
        best_momentum = 0.0
        for symbol, closes in strategy_input.closes_by_symbol.items():
            if len(closes) < required_closes:
                continue
            current_close = closes[-1]
            momentum_close = closes[-self._momentum_lookback_days - 1]
            trend_close = closes[-self._trend_lookback_days - 1]
            if momentum_close <= 0.0 or trend_close <= 0.0:
                continue
            momentum_return = (current_close / momentum_close) - 1.0
            trend_return = (current_close / trend_close) - 1.0
            if trend_return > 0.0 and momentum_return > best_momentum:
                best_symbol = symbol
                best_momentum = momentum_return
        if not best_symbol:
            return TargetWeights(target_weights={})
        return TargetWeights(target_weights={best_symbol: 1.0})
