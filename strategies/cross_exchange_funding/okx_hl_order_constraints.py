from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OkxInstrumentConstraints:
    inst_id: str
    state: str
    contract_type: str
    contract_value: Decimal
    contract_value_currency: str
    lot_size: Decimal
    min_size: Decimal
    tick_size: Decimal
    max_leverage: Decimal
    max_market_size: Decimal


@dataclass(frozen=True)
class HyperliquidInstrumentConstraints:
    asset: str
    size_decimals: int
    max_leverage: Decimal
    mark_price: Decimal
    mid_price: Decimal
    oracle_price: Decimal
    open_interest: Decimal
    day_notional_volume: Decimal
    impact_bid_price: Decimal
    impact_ask_price: Decimal


@dataclass(frozen=True)
class OkxHlOrderConstraints:
    generated_at: str
    asset: str
    paper_notional: Decimal
    okx_inst_id: str
    okx_contracts: Decimal
    okx_contracts_rounded: Decimal
    okx_notional_rounded: Decimal
    okx_min_size: Decimal
    okx_lot_size: Decimal
    okx_tick_size: Decimal
    okx_max_leverage: Decimal
    hl_size: Decimal
    hl_size_rounded: Decimal
    hl_notional_rounded: Decimal
    hl_size_decimals: int
    hl_max_leverage: Decimal
    hl_day_notional_volume: Decimal
    okx_size_valid: bool
    hl_size_valid: bool
    notes: str


def build_okx_hl_order_constraints(
    *,
    asset: str = "BTC",
    paper_notional: Decimal = Decimal("1000"),
) -> OkxHlOrderConstraints:
    generated_at = datetime.now(UTC).isoformat()
    okx = _fetch_okx_instrument(asset)
    hl = _fetch_hyperliquid_instrument(asset)
    reference_price = hl.mid_price if hl.mid_price > 0 else hl.mark_price
    okx_contracts = paper_notional / (reference_price * okx.contract_value)
    okx_contracts_rounded = _round_down_step(okx_contracts, okx.lot_size)
    okx_notional_rounded = okx_contracts_rounded * okx.contract_value * reference_price
    hl_size = paper_notional / reference_price
    hl_size_rounded = _round_down_decimals(hl_size, hl.size_decimals)
    hl_notional_rounded = hl_size_rounded * reference_price
    okx_size_valid = okx_contracts_rounded >= okx.min_size
    hl_size_valid = hl_size_rounded > 0
    return OkxHlOrderConstraints(
        generated_at=generated_at,
        asset=asset,
        paper_notional=paper_notional,
        okx_inst_id=okx.inst_id,
        okx_contracts=okx_contracts,
        okx_contracts_rounded=okx_contracts_rounded,
        okx_notional_rounded=okx_notional_rounded,
        okx_min_size=okx.min_size,
        okx_lot_size=okx.lot_size,
        okx_tick_size=okx.tick_size,
        okx_max_leverage=okx.max_leverage,
        hl_size=hl_size,
        hl_size_rounded=hl_size_rounded,
        hl_notional_rounded=hl_notional_rounded,
        hl_size_decimals=hl.size_decimals,
        hl_max_leverage=hl.max_leverage,
        hl_day_notional_volume=hl.day_notional_volume,
        okx_size_valid=okx_size_valid,
        hl_size_valid=hl_size_valid,
        notes=_notes(okx_size_valid=okx_size_valid, hl_size_valid=hl_size_valid),
    )


def write_order_constraints_csv(
    constraints: OkxHlOrderConstraints,
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
                "paper_notional",
                "okx_inst_id",
                "okx_contracts",
                "okx_contracts_rounded",
                "okx_notional_rounded",
                "okx_min_size",
                "okx_lot_size",
                "okx_tick_size",
                "okx_max_leverage",
                "hl_size",
                "hl_size_rounded",
                "hl_notional_rounded",
                "hl_size_decimals",
                "hl_max_leverage",
                "hl_day_notional_volume",
                "okx_size_valid",
                "hl_size_valid",
                "notes",
            )
        )
        writer.writerow(
            (
                constraints.generated_at,
                constraints.asset,
                _fmt(constraints.paper_notional),
                constraints.okx_inst_id,
                _fmt(constraints.okx_contracts),
                _fmt(constraints.okx_contracts_rounded),
                _fmt(constraints.okx_notional_rounded),
                _fmt(constraints.okx_min_size),
                _fmt(constraints.okx_lot_size),
                _fmt(constraints.okx_tick_size),
                _fmt(constraints.okx_max_leverage),
                _fmt(constraints.hl_size),
                _fmt(constraints.hl_size_rounded),
                _fmt(constraints.hl_notional_rounded),
                constraints.hl_size_decimals,
                _fmt(constraints.hl_max_leverage),
                _fmt(constraints.hl_day_notional_volume),
                constraints.okx_size_valid,
                constraints.hl_size_valid,
                constraints.notes,
            )
        )
    return output_path


def write_order_constraints_md(
    constraints: OkxHlOrderConstraints,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# OKX-Hyperliquid Order Constraints\n\n")
        handle.write(f"Generated: `{constraints.generated_at}`\n\n")
        handle.write("This is not an order instruction. It is a paper order-shape check.\n\n")
        handle.write("## Candidate\n\n")
        handle.write(f"- Asset: `{constraints.asset}`\n")
        handle.write(f"- Paper notional: `{_fmt(constraints.paper_notional)}` USDT\n")
        handle.write("- Long venue: `OkxSwap`\n")
        handle.write("- Short venue: `HlPerp`\n\n")
        handle.write("## OKX Leg\n\n")
        handle.write(f"- Instrument: `{constraints.okx_inst_id}`\n")
        handle.write(f"- Raw contracts: `{_fmt(constraints.okx_contracts)}`\n")
        handle.write(f"- Rounded contracts: `{_fmt(constraints.okx_contracts_rounded)}`\n")
        handle.write(f"- Rounded notional: `{_fmt(constraints.okx_notional_rounded)}` USDT\n")
        handle.write(f"- Min size: `{_fmt(constraints.okx_min_size)}`\n")
        handle.write(f"- Lot size: `{_fmt(constraints.okx_lot_size)}`\n")
        handle.write(f"- Tick size: `{_fmt(constraints.okx_tick_size)}`\n")
        handle.write(f"- Max leverage: `{_fmt(constraints.okx_max_leverage)}`\n")
        handle.write(f"- Size valid: `{constraints.okx_size_valid}`\n\n")
        handle.write("## Hyperliquid Leg\n\n")
        handle.write(f"- Raw size: `{_fmt(constraints.hl_size)}` BTC\n")
        handle.write(f"- Rounded size: `{_fmt(constraints.hl_size_rounded)}` BTC\n")
        handle.write(f"- Rounded notional: `{_fmt(constraints.hl_notional_rounded)}` USDT\n")
        handle.write(f"- Size decimals: `{constraints.hl_size_decimals}`\n")
        handle.write(f"- Max leverage: `{_fmt(constraints.hl_max_leverage)}`\n")
        handle.write(
            f"- Day notional volume: `{_fmt(constraints.hl_day_notional_volume)}` USDT\n"
        )
        handle.write(f"- Size valid: `{constraints.hl_size_valid}`\n\n")
        handle.write("## Still Unknown\n\n")
        handle.write("- Actual account access and jurisdiction from the trading environment.\n")
        handle.write("- Actual maker/taker fee tier on both venues.\n")
        handle.write("- Whether maker execution is realistic without losing the funding window.\n")
        handle.write("- Margin mode, collateral movement, liquidation buffer, and kill switch.\n")
        handle.write("- Whether the funding spread persists at order-entry time.\n\n")
        handle.write("## Notes\n\n")
        handle.write(f"{constraints.notes}\n")
    return output_path


def _fetch_okx_instrument(asset: str) -> OkxInstrumentConstraints:
    inst_id = f"{asset}-USDT-SWAP"
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={"instType": "SWAP", "instId": inst_id},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", ())
    if not rows:
        raise RuntimeError(f"OKX instrument not found: {inst_id}")
    item = rows[0]
    return OkxInstrumentConstraints(
        inst_id=str(item["instId"]),
        state=str(item["state"]),
        contract_type=str(item["ctType"]),
        contract_value=Decimal(str(item["ctVal"])),
        contract_value_currency=str(item["ctValCcy"]),
        lot_size=Decimal(str(item["lotSz"])),
        min_size=Decimal(str(item["minSz"])),
        tick_size=Decimal(str(item["tickSz"])),
        max_leverage=Decimal(str(item["lever"])),
        max_market_size=Decimal(str(item["maxMktSz"] or "0")),
    )


def _fetch_hyperliquid_instrument(asset: str) -> HyperliquidInstrumentConstraints:
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={"type": "metaAndAssetCtxs"},
        timeout=30,
    )
    response.raise_for_status()
    meta, contexts = response.json()
    for asset_meta, context in zip(meta["universe"], contexts, strict=False):
        if str(asset_meta["name"]) != asset:
            continue
        impact_prices = context.get("impactPxs") or ("0", "0")
        return HyperliquidInstrumentConstraints(
            asset=asset,
            size_decimals=int(asset_meta["szDecimals"]),
            max_leverage=Decimal(str(asset_meta.get("maxLeverage") or "0")),
            mark_price=Decimal(str(context.get("markPx") or "0")),
            mid_price=Decimal(str(context.get("midPx") or context.get("markPx") or "0")),
            oracle_price=Decimal(str(context.get("oraclePx") or "0")),
            open_interest=Decimal(str(context.get("openInterest") or "0")),
            day_notional_volume=Decimal(str(context.get("dayNtlVlm") or "0")),
            impact_bid_price=Decimal(str(impact_prices[0])),
            impact_ask_price=Decimal(str(impact_prices[1])),
        )
    raise RuntimeError(f"Hyperliquid asset not found: {asset}")


def _round_down_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _round_down_decimals(value: Decimal, decimals: int) -> Decimal:
    quant = Decimal("1").scaleb(-decimals)
    return value.quantize(quant, rounding=ROUND_DOWN)


def _notes(*, okx_size_valid: bool, hl_size_valid: bool) -> str:
    if okx_size_valid and hl_size_valid:
        return "Public instrument constraints allow the paper size shape"
    notes = []
    if not okx_size_valid:
        notes.append("OKX size is below min size")
    if not hl_size_valid:
        notes.append("Hyperliquid size rounds to zero")
    return "; ".join(notes)


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--paper-notional", type=Decimal, default=Decimal("1000"))
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "okx_hl_order_constraints.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "okx_hl_order_constraints.md",
    )
    args = parser.parse_args()

    constraints = build_okx_hl_order_constraints(
        asset=args.asset,
        paper_notional=args.paper_notional,
    )
    write_order_constraints_csv(constraints, output_path=args.csv_output_path)
    write_order_constraints_md(constraints, output_path=args.md_output_path)
    print(
        constraints.asset,
        constraints.okx_inst_id,
        f"okx_contracts={_fmt(constraints.okx_contracts_rounded)}",
        f"hl_size={_fmt(constraints.hl_size_rounded)}",
        constraints.notes,
    )


if __name__ == "__main__":
    main()
