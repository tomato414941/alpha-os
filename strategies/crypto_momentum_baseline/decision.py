from __future__ import annotations

from dataclasses import dataclass

from strategies.crypto_momentum_baseline.data import DailyClose, load_daily_closes
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
    closes_by_symbol: dict[str, list[DailyClose]] | None = None,
) -> ManualPaperDecision:
    closes = closes_by_symbol if closes_by_symbol is not None else load_daily_closes()
    timestamp = _latest_shared_timestamp(closes)
    decision_input = MomentumDecisionInput(
        closes_by_symbol={
            symbol: tuple(row.close for row in symbol_closes[-8:])
            for symbol, symbol_closes in closes.items()
        },
        current_weights={},
        equity=1.0,
    )
    target = SevenDayMomentumStrategy().decide(decision_input)
    return ManualPaperDecision(
        strategy="crypto_momentum_baseline",
        mode="manual_paper",
        timestamp=timestamp,
        target=target,
    )


def _latest_shared_timestamp(closes_by_symbol: dict[str, list[DailyClose]]) -> str:
    timestamps = {symbol_closes[-1].timestamp for symbol_closes in closes_by_symbol.values()}
    if len(timestamps) != 1:
        raise ValueError("latest timestamps are not aligned")
    return timestamps.pop()


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
