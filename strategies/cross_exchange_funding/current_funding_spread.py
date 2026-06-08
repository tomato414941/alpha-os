from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass(frozen=True)
class VenueFunding:
    venue: str
    funding_rate: float
    interval_hours: float
    next_funding_time: int

    @property
    def hourly_rate(self) -> float:
        return self.funding_rate / self.interval_hours if self.interval_hours > 0.0 else 0.0


@dataclass(frozen=True)
class FundingSpread:
    timestamp: str
    asset: str
    long_venue: str
    short_venue: str
    long_hourly_rate: float
    short_hourly_rate: float
    hourly_spread: float
    annualized_spread: float


def fetch_predicted_fundings(url: str = HYPERLIQUID_INFO_URL) -> tuple[dict[str, object], ...]:
    response = requests.post(url, json={"type": "predictedFundings"}, timeout=30)
    response.raise_for_status()
    return tuple(response.json())


def build_funding_spreads(
    payload: tuple[dict[str, object], ...],
    *,
    timestamp: str | None = None,
) -> tuple[FundingSpread, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    rows: list[FundingSpread] = []
    for asset_entry in payload:
        asset, venue_entries = asset_entry
        fundings = tuple(
            funding
            for venue_entry in venue_entries
            if (funding := _venue_funding_or_none(venue_entry)) is not None
        )
        if len(fundings) < 2:
            continue
        low = min(fundings, key=lambda funding: funding.hourly_rate)
        high = max(fundings, key=lambda funding: funding.hourly_rate)
        hourly_spread = high.hourly_rate - low.hourly_rate
        rows.append(
            FundingSpread(
                timestamp=observed_at,
                asset=str(asset),
                long_venue=low.venue,
                short_venue=high.venue,
                long_hourly_rate=low.hourly_rate,
                short_hourly_rate=high.hourly_rate,
                hourly_spread=hourly_spread,
                annualized_spread=hourly_spread * 24.0 * 365.0,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.annualized_spread, reverse=True))


def write_funding_spreads(
    rows: tuple[FundingSpread, ...],
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
                "long_venue",
                "short_venue",
                "long_hourly_rate",
                "short_hourly_rate",
                "hourly_spread",
                "annualized_spread",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.long_venue,
                    row.short_venue,
                    f"{row.long_hourly_rate:.12f}",
                    f"{row.short_hourly_rate:.12f}",
                    f"{row.hourly_spread:.12f}",
                    f"{row.annualized_spread:.8f}",
                )
            )
    return output_path


def _venue_funding_or_none(venue_entry: list[object]) -> VenueFunding | None:
    venue, details = venue_entry
    if details is None:
        return None
    if "fundingIntervalHours" not in details:
        return None
    return VenueFunding(
        venue=str(venue),
        funding_rate=float(details["fundingRate"]),
        interval_hours=float(details["fundingIntervalHours"]),
        next_funding_time=int(details["nextFundingTime"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_funding_spread.csv",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_funding_spreads(fetch_predicted_fundings())
    write_funding_spreads(rows, output_path=args.output_path)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.long_venue,
            row.short_venue,
            f"{row.hourly_spread:.8f}",
            f"{row.annualized_spread:.4f}",
        )


if __name__ == "__main__":
    main()
