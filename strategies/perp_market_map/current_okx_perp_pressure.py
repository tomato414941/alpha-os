from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log10
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"


@dataclass(frozen=True)
class OkxPerpPressureRow:
    timestamp: str
    asset: str
    inst_id: str
    last_price: float
    spread_bps: float
    funding_rate: float
    annualized_funding: float
    settled_funding_rate: float
    annualized_settled_funding: float
    premium: float
    day_volume_usd: float
    open_interest_usd: float
    open_interest_to_volume: float
    carry_side: str
    action: str
    pressure_score: float


def build_okx_perp_pressure_rows(
    *,
    timestamp: str | None = None,
    max_workers: int = 16,
    top_by_volume: int = 100,
) -> tuple[OkxPerpPressureRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    tickers = _fetch_usdt_swap_tickers()
    open_interest_by_inst_id = _fetch_open_interest_usd_by_inst_id()
    liquid_tickers = tuple(
        sorted(
            tickers,
            key=lambda row: _day_volume_usd(row),
            reverse=True,
        )[:top_by_volume]
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = tuple(
            executor.map(
                lambda ticker: _build_row(
                    ticker=ticker,
                    observed_at=observed_at,
                    open_interest_usd=open_interest_by_inst_id.get(ticker["instId"], 0.0),
                ),
                liquid_tickers,
            )
        )
    return tuple(
        sorted(
            (row for row in rows if row is not None),
            key=lambda row: row.pressure_score,
            reverse=True,
        )
    )


def write_okx_perp_pressure_rows(
    rows: tuple[OkxPerpPressureRow, ...],
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
                "inst_id",
                "last_price",
                "spread_bps",
                "funding_rate",
                "annualized_funding",
                "settled_funding_rate",
                "annualized_settled_funding",
                "premium",
                "day_volume_usd",
                "open_interest_usd",
                "open_interest_to_volume",
                "carry_side",
                "action",
                "pressure_score",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.inst_id,
                    f"{row.last_price:.12f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.funding_rate:.12f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.settled_funding_rate:.12f}",
                    f"{row.annualized_settled_funding:.8f}",
                    f"{row.premium:.12f}",
                    f"{row.day_volume_usd:.8f}",
                    f"{row.open_interest_usd:.8f}",
                    f"{row.open_interest_to_volume:.8f}",
                    row.carry_side,
                    row.action,
                    f"{row.pressure_score:.8f}",
                )
            )
    return output_path


def write_okx_perp_pressure_md(
    rows: tuple[OkxPerpPressureRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Perp Pressure\n\n")
        handle.write(
            "This maps current OKX USDT swap funding, premium, open interest, "
            "volume, and near-touch spread. It is a candidate screen, not a "
            "deployable strategy.\n\n"
        )
        handle.write(
            "| asset | action | ann funding | settled ann funding | premium | OI USD | volume USD | OI/vol | spread bps | score |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.annualized_funding:.6f} | "
                f"{row.annualized_settled_funding:.6f} | "
                f"{row.premium:.6f} | "
                f"{row.open_interest_usd:.0f} | "
                f"{row.day_volume_usd:.0f} | "
                f"{row.open_interest_to_volume:.4f} | "
                f"{row.spread_bps:.4f} | "
                f"{row.pressure_score:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "High scores mean the instrument is liquid enough to inspect and has "
            "large current funding or premium pressure. This does not include "
            "future-return labels, funding decay labels, fees, maker/taker fill "
            "probability, or liquidation data.\n"
        )
    return output_path


def _fetch_usdt_swap_tickers() -> tuple[dict[str, str], ...]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/tickers",
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    return tuple(
        item
        for item in response.json().get("data", ())
        if str(item.get("instId", "")).endswith("-USDT-SWAP")
    )


def _fetch_open_interest_usd_by_inst_id() -> dict[str, float]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/open-interest",
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    return {
        item["instId"]: float(item.get("oiUsd") or 0.0)
        for item in response.json().get("data", ())
        if str(item.get("instId", "")).endswith("-USDT-SWAP")
    }


def _build_row(
    *,
    ticker: dict[str, str],
    observed_at: str,
    open_interest_usd: float,
) -> OkxPerpPressureRow | None:
    funding = _fetch_funding(ticker["instId"])
    if funding is None:
        return None
    last_price = float(ticker.get("last") or 0.0)
    day_volume_usd = _day_volume_usd(ticker)
    funding_rate = float(funding.get("fundingRate") or 0.0)
    settled_funding_rate = float(funding.get("settFundingRate") or 0.0)
    annualized_funding = funding_rate * 3.0 * 365.0
    annualized_settled_funding = settled_funding_rate * 3.0 * 365.0
    premium = float(funding.get("premium") or 0.0)
    open_interest_to_volume = (
        open_interest_usd / day_volume_usd if day_volume_usd > 0.0 else 0.0
    )
    spread_bps = _spread_bps(ticker)
    return OkxPerpPressureRow(
        timestamp=observed_at,
        asset=ticker["instId"].removesuffix("-USDT-SWAP"),
        inst_id=ticker["instId"],
        last_price=last_price,
        spread_bps=spread_bps,
        funding_rate=funding_rate,
        annualized_funding=annualized_funding,
        settled_funding_rate=settled_funding_rate,
        annualized_settled_funding=annualized_settled_funding,
        premium=premium,
        day_volume_usd=day_volume_usd,
        open_interest_usd=open_interest_usd,
        open_interest_to_volume=open_interest_to_volume,
        carry_side=_carry_side(funding_rate),
        action=_action(funding_rate=funding_rate, premium=premium),
        pressure_score=_pressure_score(
            annualized_funding=annualized_funding,
            premium=premium,
            day_volume_usd=day_volume_usd,
            open_interest_usd=open_interest_usd,
            spread_bps=spread_bps,
        ),
    )


def _fetch_funding(inst_id: str) -> dict[str, str] | None:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/funding-rate",
        params={"instId": inst_id},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data", ())
    return data[0] if data else None


def _day_volume_usd(ticker: dict[str, str]) -> float:
    return float(ticker.get("volCcy24h") or 0.0) * float(ticker.get("last") or 0.0)


def _spread_bps(ticker: dict[str, str]) -> float:
    bid = float(ticker.get("bidPx") or 0.0)
    ask = float(ticker.get("askPx") or 0.0)
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 10000.0 if mid > 0.0 else 0.0


def _carry_side(funding_rate: float) -> str:
    if funding_rate > 0.0:
        return "short_perp_receives_funding"
    if funding_rate < 0.0:
        return "long_perp_receives_funding"
    return "flat_funding"


def _action(*, funding_rate: float, premium: float) -> str:
    if funding_rate < 0.0 and premium < 0.0:
        return "long_carry_discount_watch"
    if funding_rate > 0.0 and premium > 0.0:
        return "short_carry_premium_watch"
    if funding_rate < 0.0:
        return "long_carry_watch"
    if funding_rate > 0.0:
        return "short_carry_watch"
    return "flat_watch"


def _pressure_score(
    *,
    annualized_funding: float,
    premium: float,
    day_volume_usd: float,
    open_interest_usd: float,
    spread_bps: float,
) -> float:
    liquidity = log10(max(day_volume_usd, 1.0))
    crowding = log10(max(open_interest_usd, 1.0))
    premium_boost = 1.0 + min(abs(premium) * 500.0, 2.0)
    spread_penalty = max(spread_bps, 0.0) / 100.0
    return abs(annualized_funding) * liquidity * crowding * premium_boost - spread_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_okx_perp_pressure.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_okx_perp_pressure.md",
    )
    parser.add_argument("--top-by-volume", type=int, default=100)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_okx_perp_pressure_rows(top_by_volume=args.top_by_volume)
    write_okx_perp_pressure_rows(rows, output_path=args.output_path)
    write_okx_perp_pressure_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"ann_funding={row.annualized_funding:.4f}",
            f"premium={row.premium:.5f}",
            f"oi_usd={row.open_interest_usd:.0f}",
            f"vol_usd={row.day_volume_usd:.0f}",
            f"score={row.pressure_score:.4f}",
        )


if __name__ == "__main__":
    main()
