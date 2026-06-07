from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests

from strategies.cross_exchange_funding.current_funding_spread import (
    HYPERLIQUID_INFO_URL,
    FundingSpread,
    build_funding_spreads,
    fetch_predicted_fundings,
)


@dataclass(frozen=True)
class HyperliquidMarketContext:
    asset: str
    max_leverage: float
    open_interest: float
    day_notional_volume: float
    mid_price: float
    mark_price: float
    oracle_price: float
    impact_bid_price: float
    impact_ask_price: float

    @property
    def mark_oracle_abs_diff(self) -> float:
        return abs((self.mark_price / self.oracle_price) - 1.0) if self.oracle_price > 0.0 else 0.0

    @property
    def impact_spread(self) -> float:
        if self.mid_price <= 0.0:
            return 0.0
        return (self.impact_ask_price - self.impact_bid_price) / self.mid_price


@dataclass(frozen=True)
class FundingFeasibilityRow:
    spread: FundingSpread
    hl_involved: bool
    hl_max_leverage: float | None
    hl_open_interest: float | None
    hl_day_notional_volume: float | None
    hl_mark_oracle_abs_diff: float | None
    hl_impact_spread: float | None
    notes: str


def fetch_hyperliquid_market_contexts(
    url: str = HYPERLIQUID_INFO_URL,
) -> dict[str, HyperliquidMarketContext]:
    response = requests.post(url, json={"type": "metaAndAssetCtxs"}, timeout=30)
    response.raise_for_status()
    meta, contexts = response.json()
    rows: dict[str, HyperliquidMarketContext] = {}
    for asset_meta, context in zip(meta["universe"], contexts, strict=False):
        asset = str(asset_meta["name"])
        impact_prices = context.get("impactPxs") or (None, None)
        if impact_prices[0] is None or impact_prices[1] is None:
            continue
        rows[asset] = HyperliquidMarketContext(
            asset=asset,
            max_leverage=float(asset_meta.get("maxLeverage", 0.0)),
            open_interest=float(context.get("openInterest") or 0.0),
            day_notional_volume=float(context.get("dayNtlVlm") or 0.0),
            mid_price=float(context.get("midPx") or context.get("markPx") or 0.0),
            mark_price=float(context.get("markPx") or 0.0),
            oracle_price=float(context.get("oraclePx") or 0.0),
            impact_bid_price=float(impact_prices[0]),
            impact_ask_price=float(impact_prices[1]),
        )
    return rows


def build_feasibility_rows(
    spreads: tuple[FundingSpread, ...],
    *,
    hl_contexts: dict[str, HyperliquidMarketContext],
) -> tuple[FundingFeasibilityRow, ...]:
    rows: list[FundingFeasibilityRow] = []
    for spread in spreads:
        hl_involved = spread.long_venue == "HlPerp" or spread.short_venue == "HlPerp"
        context = hl_contexts.get(spread.asset) if hl_involved else None
        rows.append(
            FundingFeasibilityRow(
                spread=spread,
                hl_involved=hl_involved,
                hl_max_leverage=context.max_leverage if context is not None else None,
                hl_open_interest=context.open_interest if context is not None else None,
                hl_day_notional_volume=(
                    context.day_notional_volume if context is not None else None
                ),
                hl_mark_oracle_abs_diff=(
                    context.mark_oracle_abs_diff if context is not None else None
                ),
                hl_impact_spread=context.impact_spread if context is not None else None,
                notes=_notes(spread, hl_involved=hl_involved, context=context),
            )
        )
    return tuple(rows)


def write_feasibility_rows(
    rows: tuple[FundingFeasibilityRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "asset",
                "long_venue",
                "short_venue",
                "annualized_spread",
                "hl_involved",
                "hl_max_leverage",
                "hl_open_interest",
                "hl_day_notional_volume",
                "hl_mark_oracle_abs_diff",
                "hl_impact_spread",
                "notes",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.spread.timestamp,
                    row.spread.asset,
                    row.spread.long_venue,
                    row.spread.short_venue,
                    f"{row.spread.annualized_spread:.8f}",
                    row.hl_involved,
                    "" if row.hl_max_leverage is None else f"{row.hl_max_leverage:.4f}",
                    "" if row.hl_open_interest is None else f"{row.hl_open_interest:.8f}",
                    (
                        ""
                        if row.hl_day_notional_volume is None
                        else f"{row.hl_day_notional_volume:.8f}"
                    ),
                    (
                        ""
                        if row.hl_mark_oracle_abs_diff is None
                        else f"{row.hl_mark_oracle_abs_diff:.8f}"
                    ),
                    "" if row.hl_impact_spread is None else f"{row.hl_impact_spread:.8f}",
                    row.notes,
                )
            )
    return output_path


def _notes(
    spread: FundingSpread,
    *,
    hl_involved: bool,
    context: HyperliquidMarketContext | None,
) -> str:
    if not hl_involved:
        return "Hyperliquid not involved; external venue feasibility still unknown"
    if context is None:
        return "Hyperliquid involved but market context not found"
    notes = []
    if context.day_notional_volume <= 0.0:
        notes.append("missing day volume")
    if context.open_interest <= 0.0:
        notes.append("missing open interest")
    if context.impact_spread > 0.01:
        notes.append("wide impact spread")
    if context.mark_oracle_abs_diff > 0.01:
        notes.append("mark/oracle dislocation")
    return "; ".join(notes) if notes else "Hyperliquid context available"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_funding_feasibility.csv",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_feasibility_rows(
        build_funding_spreads(fetch_predicted_fundings()),
        hl_contexts=fetch_hyperliquid_market_contexts(),
    )
    write_feasibility_rows(rows, output_path=args.output_path)
    for row in rows[: args.top]:
        print(
            row.spread.asset,
            row.spread.long_venue,
            row.spread.short_venue,
            f"{row.spread.annualized_spread:.4f}",
            row.hl_involved,
            "" if row.hl_day_notional_volume is None else f"{row.hl_day_notional_volume:.0f}",
            "" if row.hl_impact_spread is None else f"{row.hl_impact_spread:.6f}",
            row.notes,
        )


if __name__ == "__main__":
    main()
