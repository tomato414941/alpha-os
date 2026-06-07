from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log10
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass(frozen=True)
class PerpMarketRow:
    timestamp: str
    asset: str
    max_leverage: float
    mark_price: float
    prev_day_price: float
    return_24h: float
    funding_rate: float
    annualized_funding: float
    open_interest: float
    open_interest_notional: float
    day_notional_volume: float
    premium: float
    mark_oracle_diff: float
    impact_spread: float
    carry_side: str
    attention_score: float


def fetch_hyperliquid_meta_and_contexts(
    url: str = HYPERLIQUID_INFO_URL,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    response = requests.post(url, json={"type": "metaAndAssetCtxs"}, timeout=30)
    response.raise_for_status()
    meta, contexts = response.json()
    return meta, contexts


def build_perp_market_rows(
    *,
    meta: dict[str, object],
    contexts: list[dict[str, object]],
    timestamp: str | None = None,
) -> tuple[PerpMarketRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    rows: list[PerpMarketRow] = []
    universe = meta["universe"]
    for asset_meta, context in zip(universe, contexts, strict=False):
        impact_prices = context.get("impactPxs") or (None, None)
        mid_price = _float(context.get("midPx") or context.get("markPx"))
        oracle_price = _float(context.get("oraclePx"))
        mark_price = _float(context.get("markPx"))
        prev_day_price = _float(context.get("prevDayPx"))
        impact_bid_price = _float(impact_prices[0])
        impact_ask_price = _float(impact_prices[1])
        funding_rate = _float(context.get("funding"))
        annualized_funding = funding_rate * 24.0 * 365.0
        day_notional_volume = _float(context.get("dayNtlVlm"))
        open_interest = _float(context.get("openInterest"))
        premium = _float(context.get("premium"))
        impact_spread = (
            (impact_ask_price - impact_bid_price) / mid_price if mid_price > 0.0 else 0.0
        )
        mark_oracle_diff = (
            (mark_price / oracle_price) - 1.0 if oracle_price > 0.0 else 0.0
        )
        return_24h = (mark_price / prev_day_price) - 1.0 if prev_day_price > 0.0 else 0.0
        rows.append(
            PerpMarketRow(
                timestamp=observed_at,
                asset=str(asset_meta["name"]),
                max_leverage=_float(asset_meta.get("maxLeverage")),
                mark_price=mark_price,
                prev_day_price=prev_day_price,
                return_24h=return_24h,
                funding_rate=funding_rate,
                annualized_funding=annualized_funding,
                open_interest=open_interest,
                open_interest_notional=open_interest * mark_price,
                day_notional_volume=day_notional_volume,
                premium=premium,
                mark_oracle_diff=mark_oracle_diff,
                impact_spread=impact_spread,
                carry_side=_carry_side(funding_rate),
                attention_score=_attention_score(
                    annualized_funding=annualized_funding,
                    day_notional_volume=day_notional_volume,
                    impact_spread=impact_spread,
                    premium=premium,
                ),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.attention_score, reverse=True))


def write_perp_market_rows(
    rows: tuple[PerpMarketRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "asset",
                "max_leverage",
                "mark_price",
                "prev_day_price",
                "return_24h",
                "funding_rate",
                "annualized_funding",
                "open_interest",
                "open_interest_notional",
                "day_notional_volume",
                "premium",
                "mark_oracle_diff",
                "impact_spread",
                "carry_side",
                "attention_score",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    f"{row.max_leverage:.4f}",
                    f"{row.mark_price:.12f}",
                    f"{row.prev_day_price:.12f}",
                    f"{row.return_24h:.8f}",
                    f"{row.funding_rate:.12f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.open_interest:.8f}",
                    f"{row.open_interest_notional:.8f}",
                    f"{row.day_notional_volume:.8f}",
                    f"{row.premium:.12f}",
                    f"{row.mark_oracle_diff:.12f}",
                    f"{row.impact_spread:.12f}",
                    row.carry_side,
                    f"{row.attention_score:.8f}",
                )
            )
    return output_path


def _float(value: object) -> float:
    return float(value or 0.0)


def _carry_side(funding_rate: float) -> str:
    if funding_rate > 0.0:
        return "short_perp_receives_funding"
    if funding_rate < 0.0:
        return "long_perp_receives_funding"
    return "flat_funding"


def _attention_score(
    *,
    annualized_funding: float,
    day_notional_volume: float,
    impact_spread: float,
    premium: float,
) -> float:
    liquidity = log10(max(day_notional_volume, 1.0))
    friction_penalty = max(impact_spread, 0.0) * 100.0
    return (abs(annualized_funding) * liquidity) + abs(premium) - friction_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    meta, contexts = fetch_hyperliquid_meta_and_contexts()
    rows = build_perp_market_rows(meta=meta, contexts=contexts)
    write_perp_market_rows(rows, output_path=args.output_path)
    for row in rows[: args.top]:
        print(
            row.asset,
            f"ann_funding={row.annualized_funding:.4f}",
            f"volume={row.day_notional_volume:.0f}",
            f"impact={row.impact_spread:.6f}",
            row.carry_side,
        )


if __name__ == "__main__":
    main()
