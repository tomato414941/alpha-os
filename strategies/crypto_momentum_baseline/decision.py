from __future__ import annotations

from dataclasses import dataclass

from strategies.crypto_momentum_baseline.data import (
    DailyMarketBar,
    load_daily_market_bars,
)
from strategies.crypto_momentum_baseline.strategy import (
    MomentumDecisionInput,
    SevenDayMomentumStrategy,
    TargetWeights,
)


@dataclass(frozen=True)
class ManualPaperDecision:
    strategy: str
    mode: str
    timestamp: str
    target: TargetWeights


def latest_manual_paper_decision(
    *,
    market_bars: tuple[DailyMarketBar, ...] | None = None,
) -> ManualPaperDecision:
    bars = market_bars if market_bars is not None else load_daily_market_bars()
    latest_bar = bars[-1]
    decision_input = MomentumDecisionInput(
        closes_by_symbol={
            symbol: tuple(bar.closes[symbol] for bar in bars[-8:])
            for symbol in latest_bar.closes
        },
        current_weights={},
        equity=1.0,
    )
    target = SevenDayMomentumStrategy().decide(decision_input)
    return ManualPaperDecision(
        strategy="crypto_momentum_baseline",
        mode="manual_paper",
        timestamp=latest_bar.timestamp,
        target=target,
    )


def main() -> None:
    decision = latest_manual_paper_decision()
    print(f"date={decision.timestamp}")
    print(f"strategy={decision.strategy}")
    print(f"mode={decision.mode}")
    print("target_weights:")
    if not decision.target.target_weights:
        print("  cash: 1.0")
        return
    for symbol, weight in sorted(decision.target.target_weights.items()):
        print(f"  {symbol}: {weight:.6f}")


if __name__ == "__main__":
    main()
