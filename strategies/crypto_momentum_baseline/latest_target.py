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
class LatestTargetSnapshot:
    strategy: str
    mode: str
    timestamp: str
    target: TargetWeights


def latest_target_snapshot(
    *,
    market_bars: tuple[DailyMarketBar, ...] | None = None,
) -> LatestTargetSnapshot:
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
    return LatestTargetSnapshot(
        strategy="crypto_momentum_baseline",
        mode="manual_paper",
        timestamp=latest_bar.timestamp,
        target=target,
    )


def main() -> None:
    snapshot = latest_target_snapshot()
    print(f"date={snapshot.timestamp}")
    print(f"strategy={snapshot.strategy}")
    print(f"mode={snapshot.mode}")
    print("target_weights:")
    if not snapshot.target.target_weights:
        print("  cash: 1.0")
        return
    for symbol, weight in sorted(snapshot.target.target_weights.items()):
        print(f"  {symbol}: {weight:.6f}")


if __name__ == "__main__":
    main()
