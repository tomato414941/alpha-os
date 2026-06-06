from __future__ import annotations

from strategies.daily_close.backtest import DailyCloseDecisionInput, TargetWeights


class RiskOnOffRotationStrategy:
    def __init__(
        self,
        *,
        risk_symbol: str = "QQQ",
        defensive_symbols: tuple[str, ...] = ("TLT", "GLD"),
        regime_symbol: str = "SPY",
        lookback_days: int = 126,
    ) -> None:
        self._risk_symbol = risk_symbol
        self._defensive_symbols = defensive_symbols
        self._regime_symbol = regime_symbol
        self._lookback_days = lookback_days

    def decide(self, strategy_input: DailyCloseDecisionInput) -> TargetWeights:
        regime_return = _lookback_return(
            strategy_input.closes_by_symbol.get(self._regime_symbol, ()),
            self._lookback_days,
        )
        if regime_return > 0.0:
            risk_return = _lookback_return(
                strategy_input.closes_by_symbol.get(self._risk_symbol, ()),
                self._lookback_days,
            )
            if risk_return > 0.0:
                return TargetWeights(target_weights={self._risk_symbol: 1.0})

        best_defensive_symbol = ""
        best_defensive_return = 0.0
        for symbol in self._defensive_symbols:
            symbol_return = _lookback_return(
                strategy_input.closes_by_symbol.get(symbol, ()),
                self._lookback_days,
            )
            if symbol_return > best_defensive_return:
                best_defensive_symbol = symbol
                best_defensive_return = symbol_return
        if not best_defensive_symbol:
            return TargetWeights(target_weights={})
        return TargetWeights(target_weights={best_defensive_symbol: 1.0})


def _lookback_return(closes: tuple[float, ...], lookback_days: int) -> float:
    if len(closes) < lookback_days + 1:
        return 0.0
    previous_close = closes[-lookback_days - 1]
    current_close = closes[-1]
    return (current_close / previous_close) - 1.0 if previous_close > 0.0 else 0.0
