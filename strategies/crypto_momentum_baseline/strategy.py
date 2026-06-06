from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MomentumDecisionInput:
    closes_by_symbol: dict[str, tuple[float, ...]]
    current_weights: dict[str, float]
    equity: float


@dataclass(frozen=True)
class TargetWeights:
    target_weights: dict[str, float]


class SevenDayMomentumStrategy:
    def decide(self, strategy_input: MomentumDecisionInput) -> TargetWeights:
        active_symbols = []
        for symbol, closes in strategy_input.closes_by_symbol.items():
            if len(closes) < 8:
                continue
            previous_close = closes[-8]
            current_close = closes[-1]
            if previous_close <= 0.0:
                continue
            if (current_close / previous_close) - 1.0 > 0.0:
                active_symbols.append(symbol)

        if not active_symbols:
            return TargetWeights(target_weights={})

        weight = 1.0 / len(active_symbols)
        return TargetWeights(
            target_weights={symbol: weight for symbol in active_symbols}
        )
