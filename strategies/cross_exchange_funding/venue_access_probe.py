from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


@dataclass(frozen=True)
class VenueAccessCheck:
    venue: str
    endpoint: str
    method: str
    url: str
    payload: dict[str, object] | None


@dataclass(frozen=True)
class VenueAccessResult:
    timestamp: str
    check: VenueAccessCheck
    status_code: int | None
    available: bool
    notes: str


CHECKS = (
    VenueAccessCheck(
        venue="Binance USD-M",
        endpoint="exchangeInfo",
        method="GET",
        url="https://fapi.binance.com/fapi/v1/exchangeInfo",
        payload=None,
    ),
    VenueAccessCheck(
        venue="Bybit linear",
        endpoint="instruments",
        method="GET",
        url="https://api.bybit.com/v5/market/instruments-info?category=linear&symbol=BTCUSDT",
        payload=None,
    ),
    VenueAccessCheck(
        venue="OKX swap",
        endpoint="instruments",
        method="GET",
        url="https://www.okx.com/api/v5/public/instruments?instType=SWAP&instId=BTC-USDT-SWAP",
        payload=None,
    ),
    VenueAccessCheck(
        venue="Hyperliquid",
        endpoint="metaAndAssetCtxs",
        method="POST",
        url="https://api.hyperliquid.xyz/info",
        payload={"type": "metaAndAssetCtxs"},
    ),
)


def run_venue_access_probe(
    *,
    timestamp: str | None = None,
) -> tuple[VenueAccessResult, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    return tuple(_run_check(check, timestamp=observed_at) for check in CHECKS)


def write_venue_access_results(
    rows: tuple[VenueAccessResult, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "venue",
                "endpoint",
                "method",
                "url",
                "status_code",
                "available",
                "notes",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.check.venue,
                    row.check.endpoint,
                    row.check.method,
                    row.check.url,
                    "" if row.status_code is None else row.status_code,
                    row.available,
                    row.notes,
                )
            )
    return output_path


def _run_check(
    check: VenueAccessCheck,
    *,
    timestamp: str,
) -> VenueAccessResult:
    try:
        if check.method == "POST":
            response = requests.post(check.url, json=check.payload, timeout=30)
        else:
            response = requests.get(check.url, timeout=30)
    except requests.RequestException as exc:
        return VenueAccessResult(
            timestamp=timestamp,
            check=check,
            status_code=None,
            available=False,
            notes=f"request failed: {type(exc).__name__}",
        )
    return VenueAccessResult(
        timestamp=timestamp,
        check=check,
        status_code=response.status_code,
        available=200 <= response.status_code < 300,
        notes=_notes(response.status_code),
    )


def _notes(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "reachable"
    if status_code == 451:
        return "location restricted"
    if status_code in {401, 403}:
        return "blocked or permission required"
    if status_code == 404:
        return "not found"
    return "unexpected status"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "venue_access.csv",
    )
    args = parser.parse_args()

    rows = run_venue_access_probe()
    write_venue_access_results(rows, output_path=args.output_path)
    for row in rows:
        print(
            row.check.venue,
            row.check.endpoint,
            row.status_code,
            row.available,
            row.notes,
        )


if __name__ == "__main__":
    main()
