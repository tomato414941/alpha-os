from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OkxInstrument:
    inst_id: str
    asset: str
    ct_val: float
    ct_val_ccy: str


@dataclass(frozen=True)
class FollowupOkxExecutionContextRow:
    timestamp: str
    priority: float
    asset: str
    source: str
    inst_id: str
    last_price: float
    annualized_funding: float | None
    volume_24h: float
    spread_bps: float | None
    near_depth_10bps_notional: float | None
    visible_depth_usage_1k: float | None
    action: str
    reason: str


def build_followup_okx_execution_context_rows(
    *,
    coverage_path: Path = ROOT / "current_followup_venue_coverage.csv",
    top: int = 40,
) -> tuple[FollowupOkxExecutionContextRow, ...]:
    instruments = _fetch_okx_instruments()
    coverage_rows = tuple(
        row for row in _read_rows(coverage_path) if row.get("okx_usdt_swap") == "True"
    )[:top]
    observed_at = datetime.now(UTC).isoformat()
    rows = tuple(
        _build_row(
            row=row,
            instrument=instruments.get(row["asset"]),
            timestamp=observed_at,
        )
        for row in coverage_rows
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_followup_okx_execution_context_csv(
    rows: tuple[FollowupOkxExecutionContextRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "priority",
                "asset",
                "source",
                "inst_id",
                "last_price",
                "annualized_funding",
                "volume_24h",
                "spread_bps",
                "near_depth_10bps_notional",
                "visible_depth_usage_1k",
                "action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    f"{row.priority:.4f}",
                    row.asset,
                    row.source,
                    row.inst_id,
                    f"{row.last_price:.12f}",
                    "" if row.annualized_funding is None else f"{row.annualized_funding:.8f}",
                    f"{row.volume_24h:.8f}",
                    "" if row.spread_bps is None else f"{row.spread_bps:.8f}",
                    (
                        ""
                        if row.near_depth_10bps_notional is None
                        else f"{row.near_depth_10bps_notional:.8f}"
                    ),
                    (
                        ""
                        if row.visible_depth_usage_1k is None
                        else f"{row.visible_depth_usage_1k:.8f}"
                    ),
                    row.action,
                    row.reason,
                )
            )
    return output_path


def write_followup_okx_execution_context_md(
    rows: tuple[FollowupOkxExecutionContextRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up OKX Execution Context\n\n")
        handle.write(
            "This joins follow-up candidates to current OKX USDT swap ticker, "
            "funding, spread, and public book depth. It is a rough venue context, "
            "not a fill model.\n\n"
        )
        handle.write(
            "| asset | inst | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.inst_id} | "
                f"{row.source} | "
                f"{row.priority:.4f} | "
                f"{'' if row.annualized_funding is None else f'{row.annualized_funding:.6f}'} | "
                f"{row.volume_24h:.0f} | "
                f"{'' if row.spread_bps is None else f'{row.spread_bps:.4f}'} | "
                f"{'' if row.near_depth_10bps_notional is None else f'{row.near_depth_10bps_notional:.0f}'} | "
                f"{'' if row.visible_depth_usage_1k is None else f'{row.visible_depth_usage_1k:.6f}'} | "
                f"{row.action} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "OKX coverage keeps OKX-only candidates visible. `okx_context_ok` only "
            "means the public venue context does not obviously block a small repeat "
            "observation; account fees, fill quality, and operational constraints "
            "are still unchecked.\n"
        )
    return output_path


def _build_row(
    *,
    row: dict[str, str],
    instrument: OkxInstrument | None,
    timestamp: str,
) -> FollowupOkxExecutionContextRow:
    if instrument is None:
        return FollowupOkxExecutionContextRow(
            timestamp=timestamp,
            priority=float(row.get("priority") or "0"),
            asset=row["asset"],
            source=row.get("source", ""),
            inst_id="",
            last_price=0.0,
            annualized_funding=None,
            volume_24h=0.0,
            spread_bps=None,
            near_depth_10bps_notional=None,
            visible_depth_usage_1k=None,
            action="missing_okx_instrument",
            reason="asset is not in current OKX USDT swap instruments",
        )
    ticker = _fetch_okx_ticker(instrument.inst_id)
    book = _fetch_okx_book(instrument.inst_id)
    funding = _fetch_okx_funding(instrument.inst_id)
    last_price = _float(ticker.get("last"))
    spread_bps = _spread_bps(book)
    near_depth = _near_depth_10bps_notional(book=book, instrument=instrument)
    visible_depth_usage = None if near_depth is None or near_depth <= 0.0 else 1_000.0 / near_depth
    action, reason = _action(
        volume_24h=_float(ticker.get("volCcy24h") or ticker.get("vol24h")),
        spread_bps=spread_bps,
        near_depth=near_depth,
        visible_depth_usage=visible_depth_usage,
    )
    return FollowupOkxExecutionContextRow(
        timestamp=timestamp,
        priority=float(row.get("priority") or "0"),
        asset=row["asset"],
        source=row.get("source", ""),
        inst_id=instrument.inst_id,
        last_price=last_price,
        annualized_funding=None if funding is None else funding * 24.0 * 365.0,
        volume_24h=_float(ticker.get("volCcy24h") or ticker.get("vol24h")),
        spread_bps=spread_bps,
        near_depth_10bps_notional=near_depth,
        visible_depth_usage_1k=visible_depth_usage,
        action=action,
        reason=reason,
    )


def _fetch_okx_instruments() -> dict[str, OkxInstrument]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    instruments: dict[str, OkxInstrument] = {}
    for row in response.json().get("data", ()):
        inst_id = str(row.get("instId") or "")
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        asset = str(row.get("uly") or "").split("-")[0].upper()
        instruments[asset] = OkxInstrument(
            inst_id=inst_id,
            asset=asset,
            ct_val=_float(row.get("ctVal")),
            ct_val_ccy=str(row.get("ctValCcy") or "").upper(),
        )
    return instruments


def _fetch_okx_ticker(inst_id: str) -> dict[str, object]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/ticker",
        params={"instId": inst_id},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", ())
    return dict(rows[0]) if rows else {}


def _fetch_okx_book(inst_id: str) -> dict[str, object]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/books",
        params={"instId": inst_id, "sz": "50"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", ())
    return dict(rows[0]) if rows else {}


def _fetch_okx_funding(inst_id: str) -> float | None:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/funding-rate",
        params={"instId": inst_id},
        timeout=30,
    )
    if response.status_code != 200:
        return None
    rows = response.json().get("data", ())
    if not rows:
        return None
    return _float(rows[0].get("fundingRate"))


def _spread_bps(book: dict[str, object]) -> float | None:
    bids = book.get("bids") or ()
    asks = book.get("asks") or ()
    if not bids or not asks:
        return None
    best_bid = _float(bids[0][0])
    best_ask = _float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    return ((best_ask - best_bid) / mid) * 10_000.0 if mid > 0.0 else None


def _near_depth_10bps_notional(
    *,
    book: dict[str, object],
    instrument: OkxInstrument,
) -> float | None:
    bids = book.get("bids") or ()
    asks = book.get("asks") or ()
    if not bids or not asks:
        return None
    best_bid = _float(bids[0][0])
    best_ask = _float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    bid_threshold = mid * (1.0 - 0.001)
    ask_threshold = mid * (1.0 + 0.001)
    bid_depth = sum(
        _level_notional(level=level, instrument=instrument)
        for level in bids
        if _float(level[0]) >= bid_threshold
    )
    ask_depth = sum(
        _level_notional(level=level, instrument=instrument)
        for level in asks
        if _float(level[0]) <= ask_threshold
    )
    return min(bid_depth, ask_depth)


def _level_notional(*, level: list[object], instrument: OkxInstrument) -> float:
    price = _float(level[0])
    size = _float(level[1])
    if instrument.ct_val_ccy == "USDT":
        return size * instrument.ct_val
    return size * instrument.ct_val * price


def _action(
    *,
    volume_24h: float,
    spread_bps: float | None,
    near_depth: float | None,
    visible_depth_usage: float | None,
) -> tuple[str, str]:
    if spread_bps is None or near_depth is None or visible_depth_usage is None:
        return "missing_okx_book_context", "could not fetch OKX public book context"
    if volume_24h < 1_000_000.0:
        return "okx_thin_volume_watch", "OKX 24h volume proxy is low for repeat observation"
    if spread_bps > 10.0:
        return "okx_wide_spread_watch", "OKX current spread is wide for a small repeat"
    if visible_depth_usage > 0.25:
        return "okx_thin_depth_watch", "1k notional uses too much visible OKX 10 bps depth"
    return "okx_context_ok", "OKX public context does not obviously block a small repeat"


def _sort_key(row: FollowupOkxExecutionContextRow) -> tuple[int, float, float]:
    action_priority = {
        "okx_context_ok": 3,
        "okx_thin_volume_watch": 2,
        "okx_wide_spread_watch": 1,
        "okx_thin_depth_watch": 1,
        "missing_okx_book_context": 0,
        "missing_okx_instrument": -1,
    }
    return (
        action_priority.get(row.action, 0),
        row.priority,
        row.volume_24h,
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-path",
        type=Path,
        default=ROOT / "current_followup_venue_coverage.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_followup_okx_execution_context.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_followup_okx_execution_context.md",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = build_followup_okx_execution_context_rows(
        coverage_path=args.coverage_path,
        top=args.top,
    )
    write_followup_okx_execution_context_csv(rows, output_path=args.output_path)
    write_followup_okx_execution_context_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"priority={row.priority:.4f}",
            f"spread={'' if row.spread_bps is None else f'{row.spread_bps:.4f}'}",
            f"depth10={'' if row.near_depth_10bps_notional is None else f'{row.near_depth_10bps_notional:.0f}'}",
        )


if __name__ == "__main__":
    main()
