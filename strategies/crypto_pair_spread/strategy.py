from __future__ import annotations

from dataclasses import dataclass

from strategies.daily_close.backtest import DailyCloseDecisionInput, TargetWeights


@dataclass(frozen=True)
class PairSpread:
    base_symbol: str
    quote_symbol: str


class ZScorePairSpreadStrategy:
    def __init__(
        self,
        *,
        pairs: tuple[PairSpread, ...],
        lookback_days: int = 60,
        entry_zscore: float = 1.5,
        gross_exposure: float = 1.0,
    ) -> None:
        self._pairs = pairs
        self._lookback_days = lookback_days
        self._entry_zscore = entry_zscore
        self._gross_exposure = gross_exposure

    def decide(self, strategy_input: DailyCloseDecisionInput) -> TargetWeights:
        best_pair: PairSpread | None = None
        best_zscore = 0.0
        for pair in self._pairs:
            zscore = _ratio_zscore(
                strategy_input.closes_by_symbol,
                pair=pair,
                lookback_days=self._lookback_days,
            )
            if abs(zscore) > abs(best_zscore):
                best_pair = pair
                best_zscore = zscore
        if best_pair is None or abs(best_zscore) < self._entry_zscore:
            return TargetWeights(target_weights={})

        half_gross = self._gross_exposure / 2.0
        if best_zscore > 0.0:
            return TargetWeights(
                target_weights={
                    best_pair.base_symbol: -half_gross,
                    best_pair.quote_symbol: half_gross,
                }
            )
        return TargetWeights(
            target_weights={
                best_pair.base_symbol: half_gross,
                best_pair.quote_symbol: -half_gross,
            }
        )


def _ratio_zscore(
    closes_by_symbol: dict[str, tuple[float, ...]],
    *,
    pair: PairSpread,
    lookback_days: int,
) -> float:
    base_closes = closes_by_symbol.get(pair.base_symbol, ())
    quote_closes = closes_by_symbol.get(pair.quote_symbol, ())
    if len(base_closes) < lookback_days + 1 or len(quote_closes) < lookback_days + 1:
        return 0.0
    ratios = tuple(
        base_close / quote_close
        for base_close, quote_close in zip(
            base_closes[-lookback_days:],
            quote_closes[-lookback_days:],
            strict=True,
        )
        if quote_close > 0.0
    )
    if len(ratios) < 2:
        return 0.0
    mean_ratio = sum(ratios) / len(ratios)
    variance = sum((ratio - mean_ratio) ** 2 for ratio in ratios) / len(ratios)
    if variance <= 0.0:
        return 0.0
    return (ratios[-1] - mean_ratio) / (variance**0.5)
