from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from strategies.crypto_momentum.data import (
    DATASET_DIR,
    DailyMarketBar,
    load_daily_market_bars,
)
from strategies.crypto_momentum.strategy import (
    MomentumDecisionInput,
    TargetWeights,
)
from strategies.crypto_momentum.variants import CURRENT_VARIANT, VARIANTS


@dataclass(frozen=True)
class LatestTargetSnapshot:
    strategy: str
    variant: str
    mode: str
    timestamp: str
    target: TargetWeights


def latest_target_snapshot(
    *,
    dataset_dir: Path = DATASET_DIR,
    market_bars: tuple[DailyMarketBar, ...] | None = None,
    variant: str = CURRENT_VARIANT,
) -> LatestTargetSnapshot:
    bars = (
        market_bars
        if market_bars is not None
        else load_daily_market_bars(dataset_dir=dataset_dir)
    )
    strategy_variant = VARIANTS[variant]
    latest_bar = bars[-1]
    decision_input = MomentumDecisionInput(
        closes_by_symbol={
            symbol: tuple(
                bar.closes[symbol] for bar in bars[-strategy_variant.lookback_days - 1 :]
            )
            for symbol in latest_bar.closes
        },
        current_weights={},
        equity=1.0,
    )
    target = strategy_variant.factory().decide(decision_input)
    return LatestTargetSnapshot(
        strategy="crypto_momentum",
        variant=variant,
        mode="manual_paper",
        timestamp=latest_bar.timestamp,
        target=target,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()

    snapshot = latest_target_snapshot(dataset_dir=args.dataset_dir)
    print(f"date={snapshot.timestamp}")
    print(f"strategy={snapshot.strategy}")
    print(f"variant={snapshot.variant}")
    print(f"mode={snapshot.mode}")
    print("target_weights:")
    if not snapshot.target.target_weights:
        print("  cash: 1.0")
        return
    for symbol, weight in sorted(snapshot.target.target_weights.items()):
        print(f"  {symbol}: {weight:.6f}")


if __name__ == "__main__":
    main()
