from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from strategies.cross_exchange_funding.current_funding_feasibility import (
    HyperliquidMarketContext,
    fetch_hyperliquid_market_contexts,
)
from strategies.cross_exchange_funding.current_funding_spread import (
    VenueFunding,
    fetch_predicted_fundings,
)


OKX_BASE_URL = "https://www.okx.com"


@dataclass(frozen=True)
class OkxInstrument:
    asset: str
    inst_id: str
    contract_value: float


@dataclass(frozen=True)
class OkxBookContext:
    best_bid: float
    best_ask: float
    top_bid_notional: float
    top_ask_notional: float

    @property
    def spread(self) -> float:
        mid = (self.best_bid + self.best_ask) / 2.0
        return (self.best_ask - self.best_bid) / mid if mid > 0.0 else 0.0


@dataclass(frozen=True)
class OkxHlFundingSpread:
    timestamp: str
    asset: str
    long_venue: str
    short_venue: str
    okx_hourly_rate: float
    hl_hourly_rate: float
    hourly_spread: float
    annualized_spread: float
    okx_spread: float
    okx_top_bid_notional: float
    okx_top_ask_notional: float
    hl_day_notional_volume: float
    hl_impact_spread: float
    rough_round_trip_cost: float
    breakeven_hold_hours: float
    net_8h_proxy: float
    net_24h_proxy: float
    capacity_proxy_notional: float
    notes: str


def fetch_okx_usdt_swap_instruments() -> dict[str, OkxInstrument]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    instruments: dict[str, OkxInstrument] = {}
    for item in response.json().get("data", ()):
        inst_id = str(item.get("instId", ""))
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        if item.get("state") not in {None, "live"}:
            continue
        asset = inst_id.removesuffix("-USDT-SWAP")
        instruments[asset] = OkxInstrument(
            asset=asset,
            inst_id=inst_id,
            contract_value=float(item.get("ctVal") or 0.0),
        )
    return instruments


def build_okx_hl_funding_spreads(
    *,
    timestamp: str | None = None,
    max_workers: int = 16,
    assets: tuple[str, ...] | None = None,
) -> tuple[OkxHlFundingSpread, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    okx_instruments = fetch_okx_usdt_swap_instruments()
    hl_fundings = _hl_fundings_by_asset(fetch_predicted_fundings())
    hl_contexts = fetch_hyperliquid_market_contexts()
    common_asset_set = set(okx_instruments) & set(hl_fundings)
    if assets is not None:
        common_asset_set &= set(assets)
    common_assets = tuple(sorted(common_asset_set))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = tuple(
            executor.map(
                lambda asset: _build_asset_row(
                    asset=asset,
                    observed_at=observed_at,
                    okx_instrument=okx_instruments[asset],
                    hl_funding=hl_fundings[asset],
                    hl_context=hl_contexts.get(asset),
                ),
                common_assets,
            )
        )
    return tuple(
        sorted(
            (row for row in rows if row is not None),
            key=lambda row: row.net_8h_proxy,
            reverse=True,
        )
    )


def write_okx_hl_funding_spreads(
    rows: tuple[OkxHlFundingSpread, ...],
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
                "okx_hourly_rate",
                "hl_hourly_rate",
                "hourly_spread",
                "annualized_spread",
                "okx_spread",
                "okx_top_bid_notional",
                "okx_top_ask_notional",
                "hl_day_notional_volume",
                "hl_impact_spread",
                "rough_round_trip_cost",
                "breakeven_hold_hours",
                "net_8h_proxy",
                "net_24h_proxy",
                "capacity_proxy_notional",
                "notes",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.long_venue,
                    row.short_venue,
                    f"{row.okx_hourly_rate:.12f}",
                    f"{row.hl_hourly_rate:.12f}",
                    f"{row.hourly_spread:.12f}",
                    f"{row.annualized_spread:.8f}",
                    f"{row.okx_spread:.8f}",
                    f"{row.okx_top_bid_notional:.8f}",
                    f"{row.okx_top_ask_notional:.8f}",
                    f"{row.hl_day_notional_volume:.8f}",
                    f"{row.hl_impact_spread:.8f}",
                    f"{row.rough_round_trip_cost:.8f}",
                    f"{row.breakeven_hold_hours:.4f}",
                    f"{row.net_8h_proxy:.8f}",
                    f"{row.net_24h_proxy:.8f}",
                    f"{row.capacity_proxy_notional:.8f}",
                    row.notes,
                )
            )
    return output_path


def _build_asset_row(
    *,
    asset: str,
    observed_at: str,
    okx_instrument: OkxInstrument,
    hl_funding: VenueFunding,
    hl_context: HyperliquidMarketContext | None,
) -> OkxHlFundingSpread | None:
    okx_funding = _fetch_okx_funding(okx_instrument)
    okx_book = _fetch_okx_book(okx_instrument)
    if okx_funding is None or okx_book is None:
        return None
    low_venue, high_venue = (
        ("OkxSwap", "HlPerp")
        if okx_funding.hourly_rate <= hl_funding.hourly_rate
        else ("HlPerp", "OkxSwap")
    )
    hourly_spread = abs(hl_funding.hourly_rate - okx_funding.hourly_rate)
    hl_day_volume = hl_context.day_notional_volume if hl_context is not None else 0.0
    hl_impact_spread = hl_context.impact_spread if hl_context is not None else 0.0
    rough_round_trip_cost = okx_book.spread + hl_impact_spread
    breakeven_hold_hours = (
        rough_round_trip_cost / hourly_spread if hourly_spread > 0.0 else 0.0
    )
    capacity_proxy_notional = min(
        okx_book.top_bid_notional,
        okx_book.top_ask_notional,
        hl_day_volume * 0.01,
    )
    return OkxHlFundingSpread(
        timestamp=observed_at,
        asset=asset,
        long_venue=low_venue,
        short_venue=high_venue,
        okx_hourly_rate=okx_funding.hourly_rate,
        hl_hourly_rate=hl_funding.hourly_rate,
        hourly_spread=hourly_spread,
        annualized_spread=hourly_spread * 24.0 * 365.0,
        okx_spread=okx_book.spread,
        okx_top_bid_notional=okx_book.top_bid_notional,
        okx_top_ask_notional=okx_book.top_ask_notional,
        hl_day_notional_volume=hl_day_volume,
        hl_impact_spread=hl_impact_spread,
        rough_round_trip_cost=rough_round_trip_cost,
        breakeven_hold_hours=breakeven_hold_hours,
        net_8h_proxy=(hourly_spread * 8.0) - rough_round_trip_cost,
        net_24h_proxy=(hourly_spread * 24.0) - rough_round_trip_cost,
        capacity_proxy_notional=capacity_proxy_notional,
        notes=_notes(okx_book=okx_book, hl_context=hl_context),
    )


def _hl_fundings_by_asset(
    payload: tuple[dict[str, object], ...],
) -> dict[str, VenueFunding]:
    rows: dict[str, VenueFunding] = {}
    for asset_entry in payload:
        asset, venue_entries = asset_entry
        for venue, details in venue_entries:
            if venue != "HlPerp" or details is None:
                continue
            rows[str(asset)] = VenueFunding(
                venue="HlPerp",
                funding_rate=float(details["fundingRate"]),
                interval_hours=float(details["fundingIntervalHours"]),
                next_funding_time=int(details["nextFundingTime"]),
            )
    return rows


def _fetch_okx_funding(instrument: OkxInstrument) -> VenueFunding | None:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/funding-rate",
        params={"instId": instrument.inst_id},
        timeout=30,
    )
    if response.status_code != 200:
        return None
    payload = response.json().get("data", ())
    if not payload:
        return None
    item = payload[0]
    funding_time = int(item["fundingTime"])
    previous_time = int(item.get("prevFundingTime") or 0)
    interval_hours = (
        (funding_time - previous_time) / 1000.0 / 60.0 / 60.0
        if previous_time > 0
        else 8.0
    )
    return VenueFunding(
        venue="OkxSwap",
        funding_rate=float(item["fundingRate"]),
        interval_hours=interval_hours,
        next_funding_time=funding_time,
    )


def _fetch_okx_book(instrument: OkxInstrument) -> OkxBookContext | None:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/books",
        params={"instId": instrument.inst_id, "sz": "50"},
        timeout=30,
    )
    if response.status_code != 200:
        return None
    payload = response.json().get("data", ())
    if not payload:
        return None
    book = payload[0]
    bids = tuple((float(price), float(size)) for price, size, *_ in book.get("bids", ()))
    asks = tuple((float(price), float(size)) for price, size, *_ in book.get("asks", ()))
    if not bids or not asks:
        return None
    return OkxBookContext(
        best_bid=bids[0][0],
        best_ask=asks[0][0],
        top_bid_notional=_top_notional(bids, contract_value=instrument.contract_value),
        top_ask_notional=_top_notional(asks, contract_value=instrument.contract_value),
    )


def _top_notional(
    levels: tuple[tuple[float, float], ...],
    *,
    contract_value: float,
) -> float:
    return sum(price * size * contract_value for price, size in levels)


def _notes(
    *,
    okx_book: OkxBookContext,
    hl_context: HyperliquidMarketContext | None,
) -> str:
    notes = []
    if okx_book.spread > 0.005:
        notes.append("wide OKX spread")
    if min(okx_book.top_bid_notional, okx_book.top_ask_notional) < 10_000.0:
        notes.append("thin OKX top-50 book")
    if hl_context is None:
        notes.append("missing Hyperliquid context")
    else:
        if hl_context.day_notional_volume < 100_000.0:
            notes.append("low Hyperliquid day volume")
        if hl_context.impact_spread > 0.005:
            notes.append("wide Hyperliquid impact spread")
    return "; ".join(notes) if notes else "OKX and Hyperliquid context available"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_okx_hl_funding_spread.csv",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--assets", nargs="+")
    args = parser.parse_args()

    assets = tuple(asset.upper() for asset in args.assets) if args.assets else None
    rows = build_okx_hl_funding_spreads(max_workers=args.max_workers, assets=assets)
    write_okx_hl_funding_spreads(rows, output_path=args.output_path)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.long_venue,
            row.short_venue,
            f"{row.annualized_spread:.4f}",
            f"okx_spread={row.okx_spread:.6f}",
            f"okx_bid_notional={row.okx_top_bid_notional:.0f}",
            f"okx_ask_notional={row.okx_top_ask_notional:.0f}",
            f"hl_volume={row.hl_day_notional_volume:.0f}",
            f"breakeven_hours={row.breakeven_hold_hours:.2f}",
            f"net_8h={row.net_8h_proxy:.6f}",
            f"capacity={row.capacity_proxy_notional:.0f}",
            row.notes,
        )


if __name__ == "__main__":
    main()
