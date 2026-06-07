from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BookFillCheck:
    venue: str
    asset: str
    side: str
    target_notional: Decimal
    best_bid: Decimal
    best_ask: Decimal
    mid_price: Decimal
    top_level_notional: Decimal
    filled_notional: Decimal
    average_fill_price: Decimal
    slippage_bps: Decimal
    levels_consumed: int
    fully_filled: bool


@dataclass(frozen=True)
class OkxHlBookDepthCheck:
    generated_at: str
    asset: str
    okx_check: BookFillCheck
    hl_check: BookFillCheck
    combined_taker_slippage_bps: Decimal
    notes: str


def build_book_depth_check(
    *,
    asset: str = "BTC",
    okx_target_notional: Decimal = Decimal("995.58645"),
    hl_target_notional: Decimal = Decimal("999.969535"),
    okx_side: str = "buy",
    hl_side: str = "sell",
) -> OkxHlBookDepthCheck:
    generated_at = datetime.now(UTC).isoformat()
    okx_check = _build_okx_check(
        asset=asset,
        target_notional=okx_target_notional,
        side=okx_side,
    )
    hl_check = _build_hyperliquid_check(
        asset=asset,
        target_notional=hl_target_notional,
        side=hl_side,
    )
    combined_slippage = okx_check.slippage_bps + hl_check.slippage_bps
    return OkxHlBookDepthCheck(
        generated_at=generated_at,
        asset=asset,
        okx_check=okx_check,
        hl_check=hl_check,
        combined_taker_slippage_bps=combined_slippage,
        notes=_notes(okx_check=okx_check, hl_check=hl_check),
    )


def write_book_depth_csv(
    check: OkxHlBookDepthCheck,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "generated_at",
                "venue",
                "asset",
                "side",
                "target_notional",
                "best_bid",
                "best_ask",
                "mid_price",
                "top_level_notional",
                "filled_notional",
                "average_fill_price",
                "slippage_bps",
                "levels_consumed",
                "fully_filled",
                "combined_taker_slippage_bps",
                "notes",
            )
        )
        for row in (check.okx_check, check.hl_check):
            writer.writerow(
                (
                    check.generated_at,
                    row.venue,
                    row.asset,
                    row.side,
                    _fmt(row.target_notional),
                    _fmt(row.best_bid),
                    _fmt(row.best_ask),
                    _fmt(row.mid_price),
                    _fmt(row.top_level_notional),
                    _fmt(row.filled_notional),
                    _fmt(row.average_fill_price),
                    _fmt(row.slippage_bps),
                    row.levels_consumed,
                    row.fully_filled,
                    _fmt(check.combined_taker_slippage_bps),
                    check.notes,
                )
            )
    return output_path


def write_book_depth_md(
    check: OkxHlBookDepthCheck,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Book Depth\n\n")
        handle.write(f"Generated: `{check.generated_at}`\n\n")
        handle.write("This is not an order instruction. It checks taker depth for the paper size.\n\n")
        handle.write(
            "| venue | side | target notional | top level notional | avg fill | slippage bps | levels | full |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in (check.okx_check, check.hl_check):
            handle.write(
                "| "
                f"{row.venue} | "
                f"{row.side} | "
                f"{_fmt(row.target_notional)} | "
                f"{_fmt(row.top_level_notional)} | "
                f"{_fmt(row.average_fill_price)} | "
                f"{_fmt(row.slippage_bps)} | "
                f"{row.levels_consumed} | "
                f"{row.fully_filled} |\n"
            )
        handle.write("\n")
        handle.write(f"- Combined taker slippage bps: `{_fmt(check.combined_taker_slippage_bps)}`\n")
        handle.write(f"- Notes: {check.notes}\n\n")
        handle.write("## Interpretation\n\n")
        handle.write(
            "This check only measures visible public book depth. It does not prove maker "
            "fill probability, account fees, post-only behavior, or whether the funding "
            "spread persists during execution.\n"
        )
    return output_path


def _build_okx_check(
    *,
    asset: str,
    target_notional: Decimal,
    side: str,
) -> BookFillCheck:
    instrument = _fetch_okx_instrument(asset)
    levels = _fetch_okx_book(asset)
    contract_value = Decimal(str(instrument["ctVal"]))
    bids = tuple(
        (Decimal(price), Decimal(size) * Decimal(price) * contract_value)
        for price, size, *_ in levels["bids"]
    )
    asks = tuple(
        (Decimal(price), Decimal(size) * Decimal(price) * contract_value)
        for price, size, *_ in levels["asks"]
    )
    return _build_fill_check(
        venue="OkxSwap",
        asset=asset,
        side=side,
        target_notional=target_notional,
        bids=bids,
        asks=asks,
    )


def _build_hyperliquid_check(
    *,
    asset: str,
    target_notional: Decimal,
    side: str,
) -> BookFillCheck:
    levels = _fetch_hyperliquid_book(asset)
    bid_levels, ask_levels = levels["levels"]
    bids = tuple(
        (Decimal(str(level["px"])), Decimal(str(level["px"])) * Decimal(str(level["sz"])))
        for level in bid_levels
    )
    asks = tuple(
        (Decimal(str(level["px"])), Decimal(str(level["px"])) * Decimal(str(level["sz"])))
        for level in ask_levels
    )
    return _build_fill_check(
        venue="HlPerp",
        asset=asset,
        side=side,
        target_notional=target_notional,
        bids=bids,
        asks=asks,
    )


def _build_fill_check(
    *,
    venue: str,
    asset: str,
    side: str,
    target_notional: Decimal,
    bids: tuple[tuple[Decimal, Decimal], ...],
    asks: tuple[tuple[Decimal, Decimal], ...],
) -> BookFillCheck:
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / Decimal("2")
    fill_levels = asks if side == "buy" else bids
    filled_notional, average_price, levels_consumed = _consume_levels(
        fill_levels,
        target_notional=target_notional,
    )
    if side == "buy":
        slippage_bps = ((average_price / mid_price) - Decimal("1")) * Decimal("10000")
    else:
        slippage_bps = ((mid_price / average_price) - Decimal("1")) * Decimal("10000")
    return BookFillCheck(
        venue=venue,
        asset=asset,
        side=side,
        target_notional=target_notional,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        top_level_notional=fill_levels[0][1],
        filled_notional=filled_notional,
        average_fill_price=average_price,
        slippage_bps=slippage_bps,
        levels_consumed=levels_consumed,
        fully_filled=filled_notional >= target_notional,
    )


def _consume_levels(
    levels: tuple[tuple[Decimal, Decimal], ...],
    *,
    target_notional: Decimal,
) -> tuple[Decimal, Decimal, int]:
    remaining = target_notional
    filled_notional = Decimal("0")
    base_quantity = Decimal("0")
    levels_consumed = 0
    for price, level_notional in levels:
        if remaining <= 0:
            break
        used_notional = min(remaining, level_notional)
        filled_notional += used_notional
        base_quantity += used_notional / price
        remaining -= used_notional
        levels_consumed += 1
    average_price = filled_notional / base_quantity if base_quantity > 0 else Decimal("0")
    return filled_notional, average_price, levels_consumed


def _fetch_okx_instrument(asset: str) -> dict[str, object]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={"instType": "SWAP", "instId": f"{asset}-USDT-SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", ())
    if not rows:
        raise RuntimeError(f"OKX instrument not found: {asset}")
    return rows[0]


def _fetch_okx_book(asset: str) -> dict[str, object]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/books",
        params={"instId": f"{asset}-USDT-SWAP", "sz": "50"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", ())
    if not rows:
        raise RuntimeError(f"OKX book not found: {asset}")
    return rows[0]


def _fetch_hyperliquid_book(asset: str) -> dict[str, object]:
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={"type": "l2Book", "coin": asset},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _notes(*, okx_check: BookFillCheck, hl_check: BookFillCheck) -> str:
    if not okx_check.fully_filled or not hl_check.fully_filled:
        return "paper size does not fully fill in visible top book"
    if okx_check.levels_consumed == 1 and hl_check.levels_consumed == 1:
        return "paper size fits inside the top visible level on both venues"
    return "paper size fills, but consumes multiple visible levels"


def _fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.00000001")).normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--okx-target-notional", type=Decimal, default=Decimal("995.58645"))
    parser.add_argument("--hl-target-notional", type=Decimal, default=Decimal("999.969535"))
    parser.add_argument("--okx-side", choices=("buy", "sell"), default="buy")
    parser.add_argument("--hl-side", choices=("buy", "sell"), default="sell")
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_book_depth.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_book_depth.md",
    )
    args = parser.parse_args()

    check = build_book_depth_check(
        asset=args.asset,
        okx_target_notional=args.okx_target_notional,
        hl_target_notional=args.hl_target_notional,
        okx_side=args.okx_side,
        hl_side=args.hl_side,
    )
    write_book_depth_csv(check, output_path=args.csv_output_path)
    write_book_depth_md(check, output_path=args.md_output_path)
    print(
        check.asset,
        f"okx_slippage_bps={_fmt(check.okx_check.slippage_bps)}",
        f"hl_slippage_bps={_fmt(check.hl_check.slippage_bps)}",
        f"combined_bps={_fmt(check.combined_taker_slippage_bps)}",
        check.notes,
    )


if __name__ == "__main__":
    main()
