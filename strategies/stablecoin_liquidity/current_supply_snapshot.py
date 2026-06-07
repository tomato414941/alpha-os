from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


DEFILLAMA_STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoins"


@dataclass(frozen=True)
class StablecoinSupplyRow:
    timestamp: str
    symbol: str
    name: str
    peg_type: str
    peg_mechanism: str
    current_supply_usd: float
    day_change_usd: float
    week_change_usd: float
    month_change_usd: float
    price: float


def fetch_stablecoins(url: str = DEFILLAMA_STABLECOINS_URL) -> dict[str, object]:
    response = requests.get(url, params={"includePrices": "true"}, timeout=30)
    response.raise_for_status()
    return response.json()


def build_stablecoin_supply_rows(
    payload: dict[str, object],
    *,
    timestamp: str | None = None,
) -> tuple[StablecoinSupplyRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    rows: list[StablecoinSupplyRow] = []
    for asset in payload.get("peggedAssets") or ():
        current = _pegged_usd(asset.get("circulating"))
        prev_day = _pegged_usd(asset.get("circulatingPrevDay"))
        prev_week = _pegged_usd(asset.get("circulatingPrevWeek"))
        prev_month = _pegged_usd(asset.get("circulatingPrevMonth"))
        if current <= 0.0:
            continue
        rows.append(
            StablecoinSupplyRow(
                timestamp=observed_at,
                symbol=str(asset.get("symbol") or ""),
                name=str(asset.get("name") or ""),
                peg_type=str(asset.get("pegType") or ""),
                peg_mechanism=str(asset.get("pegMechanism") or ""),
                current_supply_usd=current,
                day_change_usd=current - prev_day,
                week_change_usd=current - prev_week,
                month_change_usd=current - prev_month,
                price=float(asset.get("price") or 0.0),
            )
        )
    return tuple(sorted(rows, key=lambda row: abs(row.week_change_usd), reverse=True))


def write_stablecoin_supply_rows(
    rows: tuple[StablecoinSupplyRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "symbol",
                "name",
                "peg_type",
                "peg_mechanism",
                "current_supply_usd",
                "day_change_usd",
                "week_change_usd",
                "month_change_usd",
                "price",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.symbol,
                    row.name,
                    row.peg_type,
                    row.peg_mechanism,
                    f"{row.current_supply_usd:.2f}",
                    f"{row.day_change_usd:.2f}",
                    f"{row.week_change_usd:.2f}",
                    f"{row.month_change_usd:.2f}",
                    f"{row.price:.8f}",
                )
            )
    return output_path


def _pegged_usd(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    return float(value.get("peggedUSD") or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_supply_snapshot.csv",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_stablecoin_supply_rows(fetch_stablecoins())
    write_stablecoin_supply_rows(rows, output_path=args.output_path)
    for row in rows[: args.top]:
        print(
            row.symbol,
            row.name,
            f"supply={row.current_supply_usd:.0f}",
            f"week_change={row.week_change_usd:.0f}",
            f"price={row.price:.6f}",
        )


if __name__ == "__main__":
    main()

