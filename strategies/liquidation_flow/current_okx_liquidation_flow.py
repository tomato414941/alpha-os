from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import log10, sqrt
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"


@dataclass(frozen=True)
class OkxInstrument:
    asset: str
    inst_family: str
    contract_value: float


@dataclass(frozen=True)
class LiquidationFlowRow:
    timestamp: str
    asset: str
    inst_family: str
    observations: int
    latest_liquidation_at: str
    long_liquidation_notional: float
    short_liquidation_notional: float
    total_liquidation_notional: float
    liquidation_to_volume: float
    forced_buy_sell_imbalance: float
    day_volume_usd: float
    action: str
    cascade_score: float


def build_okx_liquidation_flow_rows(
    *,
    timestamp: str | None = None,
    lookback_minutes: int = 60,
    top_by_volume: int = 30,
    max_workers: int = 4,
) -> tuple[LiquidationFlowRow, ...]:
    observed_at_dt = datetime.now(UTC)
    observed_at = timestamp or observed_at_dt.isoformat()
    cutoff_ms = int((observed_at_dt - timedelta(minutes=lookback_minutes)).timestamp() * 1000)
    instruments = _fetch_usdt_swap_instruments()
    tickers = _fetch_usdt_swap_tickers()
    volume_by_asset = {ticker["asset"]: ticker["day_volume_usd"] for ticker in tickers}
    liquid_assets = tuple(
        row["asset"]
        for row in sorted(tickers, key=lambda item: item["day_volume_usd"], reverse=True)[
            :top_by_volume
        ]
        if row["asset"] in instruments
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = tuple(
            executor.map(
                lambda asset: _build_asset_row(
                    instrument=instruments[asset],
                    observed_at=observed_at,
                    cutoff_ms=cutoff_ms,
                    day_volume_usd=volume_by_asset.get(asset, 0.0),
                ),
                liquid_assets,
            )
        )
    return tuple(
        sorted(
            (row for row in rows if row is not None),
            key=lambda row: row.cascade_score,
            reverse=True,
        )
    )


def write_liquidation_flow_rows(
    rows: tuple[LiquidationFlowRow, ...],
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
                "inst_family",
                "observations",
                "latest_liquidation_at",
                "long_liquidation_notional",
                "short_liquidation_notional",
                "total_liquidation_notional",
                "liquidation_to_volume",
                "forced_buy_sell_imbalance",
                "day_volume_usd",
                "action",
                "cascade_score",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.inst_family,
                    row.observations,
                    row.latest_liquidation_at,
                    f"{row.long_liquidation_notional:.8f}",
                    f"{row.short_liquidation_notional:.8f}",
                    f"{row.total_liquidation_notional:.8f}",
                    f"{row.liquidation_to_volume:.8f}",
                    f"{row.forced_buy_sell_imbalance:.8f}",
                    f"{row.day_volume_usd:.8f}",
                    row.action,
                    f"{row.cascade_score:.8f}",
                )
            )
    return output_path


def write_liquidation_flow_md(
    rows: tuple[LiquidationFlowRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Flow\n\n")
        handle.write(
            "This maps recent OKX USDT swap liquidation flow. Long liquidation "
            "means forced sell flow; short liquidation means forced buy flow.\n\n"
        )
        handle.write(
            "| asset | action | obs | long liq USD | short liq USD | total liq USD | liq/vol | imbalance | score |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.observations} | "
                f"{row.long_liquidation_notional:.0f} | "
                f"{row.short_liquidation_notional:.0f} | "
                f"{row.total_liquidation_notional:.0f} | "
                f"{row.liquidation_to_volume:.6f} | "
                f"{row.forced_buy_sell_imbalance:.6f} | "
                f"{row.cascade_score:.6f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This is a live event-flow screen. It does not prove whether the "
            "right trade is continuation, reversal, or no trade. The next test "
            "is to label post-liquidation returns and join with funding, open "
            "interest, and order-book depth.\n"
        )
    return output_path


def _fetch_usdt_swap_instruments() -> dict[str, OkxInstrument]:
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
        asset = inst_id.removesuffix("-USDT-SWAP")
        instruments[asset] = OkxInstrument(
            asset=asset,
            inst_family=str(item.get("instFamily") or f"{asset}-USDT"),
            contract_value=float(item.get("ctVal") or 0.0),
        )
    return instruments


def _fetch_usdt_swap_tickers() -> tuple[dict[str, float | str], ...]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/tickers",
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    rows: list[dict[str, float | str]] = []
    for item in response.json().get("data", ()):
        inst_id = str(item.get("instId", ""))
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        last = float(item.get("last") or 0.0)
        rows.append(
            {
                "asset": inst_id.removesuffix("-USDT-SWAP"),
                "day_volume_usd": float(item.get("volCcy24h") or 0.0) * last,
            }
        )
    return tuple(rows)


def _build_asset_row(
    *,
    instrument: OkxInstrument,
    observed_at: str,
    cutoff_ms: int,
    day_volume_usd: float,
) -> LiquidationFlowRow | None:
    details = tuple(
        detail
        for detail in _fetch_liquidation_details(instrument.inst_family)
        if int(detail.get("ts") or detail.get("time") or 0) >= cutoff_ms
    )
    if not details:
        return None
    long_liquidation_notional = sum(
        _detail_notional(detail, instrument=instrument)
        for detail in details
        if detail.get("posSide") == "long"
    )
    short_liquidation_notional = sum(
        _detail_notional(detail, instrument=instrument)
        for detail in details
        if detail.get("posSide") == "short"
    )
    total_liquidation_notional = long_liquidation_notional + short_liquidation_notional
    if total_liquidation_notional <= 0.0:
        return None
    imbalance = (short_liquidation_notional - long_liquidation_notional) / (
        total_liquidation_notional
    )
    latest_ms = max(int(detail.get("ts") or detail.get("time") or 0) for detail in details)
    liquidation_to_volume = (
        total_liquidation_notional / day_volume_usd if day_volume_usd > 0.0 else 0.0
    )
    return LiquidationFlowRow(
        timestamp=observed_at,
        asset=instrument.asset,
        inst_family=instrument.inst_family,
        observations=len(details),
        latest_liquidation_at=datetime.fromtimestamp(latest_ms / 1000.0, UTC).isoformat(),
        long_liquidation_notional=long_liquidation_notional,
        short_liquidation_notional=short_liquidation_notional,
        total_liquidation_notional=total_liquidation_notional,
        liquidation_to_volume=liquidation_to_volume,
        forced_buy_sell_imbalance=imbalance,
        day_volume_usd=day_volume_usd,
        action=_action(imbalance),
        cascade_score=_cascade_score(
            total_liquidation_notional=total_liquidation_notional,
            liquidation_to_volume=liquidation_to_volume,
            imbalance=imbalance,
        ),
    )


def _fetch_liquidation_details(inst_family: str) -> tuple[dict[str, str], ...]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/liquidation-orders",
        params={
            "instType": "SWAP",
            "mgnMode": "cross",
            "instFamily": inst_family,
            "state": "filled",
        },
        timeout=30,
    )
    if response.status_code == 429:
        return ()
    response.raise_for_status()
    details: list[dict[str, str]] = []
    for item in response.json().get("data", ()):
        details.extend(item.get("details", ()))
    return tuple(details)


def _detail_notional(detail: dict[str, str], *, instrument: OkxInstrument) -> float:
    size = float(detail.get("sz") or 0.0)
    bankruptcy_price = float(detail.get("bkPx") or 0.0)
    return size * instrument.contract_value * bankruptcy_price


def _action(imbalance: float) -> str:
    if imbalance < -0.5:
        return "long_liquidation_cascade_watch"
    if imbalance > 0.5:
        return "short_liquidation_squeeze_watch"
    return "mixed_liquidation_flow_watch"


def _cascade_score(
    *,
    total_liquidation_notional: float,
    liquidation_to_volume: float,
    imbalance: float,
) -> float:
    return (
        log10(max(total_liquidation_notional, 1.0))
        * sqrt(max(liquidation_to_volume, 0.0))
        * abs(imbalance)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_okx_liquidation_flow.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_okx_liquidation_flow.md",
    )
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--top-by-volume", type=int, default=30)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_okx_liquidation_flow_rows(
        lookback_minutes=args.lookback_minutes,
        top_by_volume=args.top_by_volume,
    )
    write_liquidation_flow_rows(rows, output_path=args.output_path)
    write_liquidation_flow_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"obs={row.observations}",
            f"total_liq={row.total_liquidation_notional:.0f}",
            f"imbalance={row.forced_buy_sell_imbalance:.4f}",
            f"score={row.cascade_score:.4f}",
        )


if __name__ == "__main__":
    main()
