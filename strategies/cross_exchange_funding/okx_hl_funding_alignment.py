from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FundingAlignment:
    generated_at: str
    asset: str
    okx_inst_id: str
    okx_funding_rate: Decimal
    okx_settled_funding_rate: Decimal
    okx_first_funding_time: str
    okx_next_funding_time: str
    okx_interval_hours: Decimal
    hl_funding_rate: Decimal
    hl_first_funding_time: str
    hl_interval_hours: Decimal
    first_event_gap_hours: Decimal
    okx_events_8h: int
    hl_events_8h: int
    okx_events_24h: int
    hl_events_24h: int
    okx_long_expected_rate_per_event: Decimal
    hl_short_expected_rate_per_event: Decimal
    notes: str


def build_funding_alignment(asset: str = "BTC") -> FundingAlignment:
    now = datetime.now(UTC)
    okx = _fetch_okx_funding(asset)
    hl = _fetch_hyperliquid_funding(asset)
    okx_interval = _hours_between(okx["previous_funding_time"], okx["funding_time"])
    hl_interval = Decimal(str(hl["interval_hours"]))
    okx_first_time = _first_future_time(
        now,
        (
            okx["funding_time"],
            okx["next_funding_time"],
        ),
        interval_hours=okx_interval,
    )
    okx_next_time = max(okx["funding_time"], okx["next_funding_time"])
    hl_first_time = _first_future_time(
        now,
        (hl["next_funding_time"],),
        interval_hours=hl_interval,
    )
    return FundingAlignment(
        generated_at=now.isoformat(),
        asset=asset,
        okx_inst_id=f"{asset}-USDT-SWAP",
        okx_funding_rate=okx["funding_rate"],
        okx_settled_funding_rate=okx["settled_funding_rate"],
        okx_first_funding_time=okx_first_time.isoformat(),
        okx_next_funding_time=okx_next_time.isoformat(),
        okx_interval_hours=okx_interval,
        hl_funding_rate=hl["funding_rate"],
        hl_first_funding_time=hl_first_time.isoformat(),
        hl_interval_hours=hl_interval,
        first_event_gap_hours=abs(_hours_between(okx_first_time, hl_first_time)),
        okx_events_8h=_event_count(
            start=now,
            first_event=okx_first_time,
            interval_hours=okx_interval,
            horizon_hours=Decimal("8"),
        ),
        hl_events_8h=_event_count(
            start=now,
            first_event=hl_first_time,
            interval_hours=hl_interval,
            horizon_hours=Decimal("8"),
        ),
        okx_events_24h=_event_count(
            start=now,
            first_event=okx_first_time,
            interval_hours=okx_interval,
            horizon_hours=Decimal("24"),
        ),
        hl_events_24h=_event_count(
            start=now,
            first_event=hl_first_time,
            interval_hours=hl_interval,
            horizon_hours=Decimal("24"),
        ),
        okx_long_expected_rate_per_event=-okx["funding_rate"],
        hl_short_expected_rate_per_event=hl["funding_rate"],
        notes=_notes(okx_funding_rate=okx["funding_rate"], hl_funding_rate=hl["funding_rate"]),
    )


def write_funding_alignment_csv(
    alignment: FundingAlignment,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "generated_at",
                "asset",
                "okx_inst_id",
                "okx_funding_rate",
                "okx_settled_funding_rate",
                "okx_first_funding_time",
                "okx_next_funding_time",
                "okx_interval_hours",
                "hl_funding_rate",
                "hl_first_funding_time",
                "hl_interval_hours",
                "first_event_gap_hours",
                "okx_events_8h",
                "hl_events_8h",
                "okx_events_24h",
                "hl_events_24h",
                "okx_long_expected_rate_per_event",
                "hl_short_expected_rate_per_event",
                "notes",
            )
        )
        writer.writerow(
            (
                alignment.generated_at,
                alignment.asset,
                alignment.okx_inst_id,
                _fmt(alignment.okx_funding_rate),
                _fmt(alignment.okx_settled_funding_rate),
                alignment.okx_first_funding_time,
                alignment.okx_next_funding_time,
                _fmt(alignment.okx_interval_hours),
                _fmt(alignment.hl_funding_rate),
                alignment.hl_first_funding_time,
                _fmt(alignment.hl_interval_hours),
                _fmt(alignment.first_event_gap_hours),
                alignment.okx_events_8h,
                alignment.hl_events_8h,
                alignment.okx_events_24h,
                alignment.hl_events_24h,
                _fmt(alignment.okx_long_expected_rate_per_event),
                _fmt(alignment.hl_short_expected_rate_per_event),
                alignment.notes,
            )
        )
    return output_path


def write_funding_alignment_md(
    alignment: FundingAlignment,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Funding Alignment\n\n")
        handle.write(f"Generated: `{alignment.generated_at}`\n\n")
        handle.write("This is not a trade instruction. It checks funding timestamp alignment.\n\n")
        handle.write("## Candidate\n\n")
        handle.write(f"- Asset: `{alignment.asset}`\n")
        handle.write("- Long venue: `OkxSwap`\n")
        handle.write("- Short venue: `HlPerp`\n\n")
        handle.write("## Funding Times\n\n")
        handle.write(f"- OKX instrument: `{alignment.okx_inst_id}`\n")
        handle.write(f"- OKX current funding rate: `{_fmt(alignment.okx_funding_rate)}`\n")
        handle.write(
            f"- OKX long expected rate per event: `{_fmt(alignment.okx_long_expected_rate_per_event)}`\n"
        )
        handle.write(f"- OKX first funding time: `{alignment.okx_first_funding_time}`\n")
        handle.write(f"- OKX interval hours: `{_fmt(alignment.okx_interval_hours)}`\n")
        handle.write(f"- Hyperliquid funding rate: `{_fmt(alignment.hl_funding_rate)}`\n")
        handle.write(
            f"- Hyperliquid short expected rate per event: `{_fmt(alignment.hl_short_expected_rate_per_event)}`\n"
        )
        handle.write(f"- Hyperliquid first funding time: `{alignment.hl_first_funding_time}`\n")
        handle.write(f"- Hyperliquid interval hours: `{_fmt(alignment.hl_interval_hours)}`\n")
        handle.write(f"- First event gap hours: `{_fmt(alignment.first_event_gap_hours)}`\n\n")
        handle.write("## Event Counts\n\n")
        handle.write(f"- OKX events within 8h: `{alignment.okx_events_8h}`\n")
        handle.write(f"- Hyperliquid events within 8h: `{alignment.hl_events_8h}`\n")
        handle.write(f"- OKX events within 24h: `{alignment.okx_events_24h}`\n")
        handle.write(f"- Hyperliquid events within 24h: `{alignment.hl_events_24h}`\n\n")
        handle.write("## Notes\n\n")
        handle.write(f"{alignment.notes}\n\n")
        handle.write("## Still Unknown\n\n")
        handle.write("- Whether these rates persist until each funding event.\n")
        handle.write("- Whether entry can be completed before the relevant funding windows.\n")
        handle.write("- Exact account fee tier and collateral/margin state.\n")
    return output_path


def _fetch_okx_funding(asset: str) -> dict[str, Decimal | datetime]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/funding-rate",
        params={"instId": f"{asset}-USDT-SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", ())
    if not rows:
        raise RuntimeError(f"OKX funding not found for {asset}")
    item = rows[0]
    return {
        "funding_rate": Decimal(str(item["fundingRate"])),
        "settled_funding_rate": Decimal(str(item.get("settFundingRate") or "0")),
        "funding_time": _ms_to_datetime(int(item["fundingTime"])),
        "next_funding_time": _ms_to_datetime(int(item["nextFundingTime"])),
        "previous_funding_time": _ms_to_datetime(int(item["prevFundingTime"])),
    }


def _fetch_hyperliquid_funding(asset: str) -> dict[str, Decimal | datetime]:
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={"type": "predictedFundings"},
        timeout=30,
    )
    response.raise_for_status()
    for payload_asset, venue_entries in response.json():
        if payload_asset != asset:
            continue
        for venue, details in venue_entries:
            if venue != "HlPerp" or details is None:
                continue
            return {
                "funding_rate": Decimal(str(details["fundingRate"])),
                "next_funding_time": _ms_to_datetime(int(details["nextFundingTime"])),
                "interval_hours": Decimal(str(details["fundingIntervalHours"])),
            }
    raise RuntimeError(f"Hyperliquid funding not found for {asset}")


def _first_future_time(
    now: datetime,
    candidates: tuple[datetime, ...],
    *,
    interval_hours: Decimal,
) -> datetime:
    future = tuple(candidate for candidate in candidates if candidate > now)
    if future:
        return min(future)
    candidate = max(candidates)
    while candidate <= now and interval_hours > 0:
        candidate += timedelta(hours=float(interval_hours))
    return candidate


def _event_count(
    *,
    start: datetime,
    first_event: datetime,
    interval_hours: Decimal,
    horizon_hours: Decimal,
) -> int:
    if interval_hours <= 0:
        return 0
    end = start + timedelta(hours=float(horizon_hours))
    count = 0
    event_time = first_event
    while event_time <= end:
        if event_time > start:
            count += 1
        event_time += timedelta(hours=float(interval_hours))
    return count


def _hours_between(left: datetime, right: datetime) -> Decimal:
    seconds = Decimal(str((right - left).total_seconds()))
    return seconds / Decimal("3600")


def _ms_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def _notes(*, okx_funding_rate: Decimal, hl_funding_rate: Decimal) -> str:
    if okx_funding_rate < 0 and hl_funding_rate > 0:
        return "Current signs match the paper direction: long OKX and short Hyperliquid both expect funding income"
    return "Current funding signs do not both support the paper direction"


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTC")
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_funding_alignment.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_funding_alignment.md",
    )
    args = parser.parse_args()

    alignment = build_funding_alignment(asset=args.asset)
    write_funding_alignment_csv(alignment, output_path=args.csv_output_path)
    write_funding_alignment_md(alignment, output_path=args.md_output_path)
    print(
        alignment.asset,
        f"okx_first={alignment.okx_first_funding_time}",
        f"hl_first={alignment.hl_first_funding_time}",
        f"gap_h={_fmt(alignment.first_event_gap_hours)}",
        alignment.notes,
    )


if __name__ == "__main__":
    main()
