from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


OKX_BASE_URL = "https://www.okx.com"
ROOT = Path(__file__).resolve().parent
DEFAULT_SIZES_USD = (100.0, 250.0, 500.0, 1_000.0)


@dataclass(frozen=True)
class LiquidationIntensityPaperGateRow:
    asset: str
    action: str
    label_status: str
    trade_direction: str
    candidate_size_usd: float
    label_bps_15m: float
    spread_bps: float
    depth_10bps_notional: float
    visible_depth_usage: float
    conservative_cost_bps: float
    conservative_net_bps: float
    gate_action: str
    reason: str
    next_step: str


def build_liquidation_intensity_paper_gate_rows(
    *,
    label_path: Path = ROOT / "current_okx_liquidation_intensity_forward_labels.csv",
    sizes_usd: tuple[float, ...] = DEFAULT_SIZES_USD,
    fee_bps_per_fill: float = 5.0,
) -> tuple[LiquidationIntensityPaperGateRow, ...]:
    labels = tuple(
        row
        for row in _read_rows(label_path)
        if row.get("label_status")
        in {
            "continuation_15m_1h_supported",
            "reversal_15m_1h_supported",
            "continuation_15m_supported_pending_1h",
            "reversal_15m_supported_pending_1h",
        }
    )
    contract_values = _fetch_usdt_swap_instruments()
    books = {
        asset: _fetch_okx_book(f"{asset}-USDT-SWAP")
        for asset in sorted({row.get("asset", "") for row in labels if row.get("asset")})
    }
    rows = tuple(
        _build_row(
            label=label,
            size_usd=size,
            fee_bps_per_fill=fee_bps_per_fill,
            contract_value=contract_values.get(label.get("asset", ""), 0.0),
            book=books.get(label.get("asset", ""), {"bids": [], "asks": []}),
        )
        for label in labels
        for size in sizes_usd
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.gate_action in {"small_paper_probe", "small_paper_probe_pending_1h"},
                row.conservative_net_bps,
                -row.visible_depth_usage,
            ),
            reverse=True,
        )
    )


def write_liquidation_intensity_paper_gate_rows(
    rows: tuple[LiquidationIntensityPaperGateRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "action",
                "label_status",
                "trade_direction",
                "candidate_size_usd",
                "label_bps_15m",
                "spread_bps",
                "depth_10bps_notional",
                "visible_depth_usage",
                "conservative_cost_bps",
                "conservative_net_bps",
                "gate_action",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.label_status,
                    row.trade_direction,
                    f"{row.candidate_size_usd:.2f}",
                    f"{row.label_bps_15m:.8f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.depth_10bps_notional:.8f}",
                    f"{row.visible_depth_usage:.8f}",
                    f"{row.conservative_cost_bps:.8f}",
                    f"{row.conservative_net_bps:.8f}",
                    row.gate_action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_liquidation_intensity_paper_gate_md(
    rows: tuple[LiquidationIntensityPaperGateRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Intensity Paper Gate\n\n")
        handle.write(
            "This applies a rough OKX spread, taker-fee, and visible-depth haircut "
            "to liquidation-intensity forward labels. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | label | side | size USD | label bps | cost bps | net bps | depth 10bps USD | usage | gate | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.label_status} | "
                f"{row.trade_direction} | "
                f"{row.candidate_size_usd:.0f} | "
                f"{row.label_bps_15m:.2f} | "
                f"{row.conservative_cost_bps:.2f} | "
                f"{row.conservative_net_bps:.2f} | "
                f"{row.depth_10bps_notional:.0f} | "
                f"{row.visible_depth_usage:.4f} | "
                f"{row.gate_action} | "
                f"{row.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`small_paper_probe` means a label with 1h support still survives this rough gate. "
            "`small_paper_probe_pending_1h` means only the 15m label is mature. Both still need "
            "real fills, funding PnL, stop behavior, and repeat-event evidence.\n"
        )
    return output_path


def _build_row(
    *,
    label: dict[str, str],
    size_usd: float,
    fee_bps_per_fill: float,
    contract_value: float,
    book: dict[str, list[list[str]]],
) -> LiquidationIntensityPaperGateRow:
    asset = label.get("asset", "")
    label_status = label.get("label_status", "")
    trade_direction = _trade_direction(label)
    label_bps = _label_bps(label) * 10_000.0
    spread_bps, depth_10bps = _book_context(
        book=book,
        contract_value=contract_value,
        trade_direction=trade_direction,
    )
    visible_depth_usage = size_usd / depth_10bps if depth_10bps > 0.0 else 1_000_000.0
    conservative_cost = (fee_bps_per_fill * 2.0) + spread_bps + (min(visible_depth_usage, 1.0) * 10.0)
    conservative_net = label_bps - conservative_cost
    gate_action, reason = _gate_action(
        label_status=label_status,
        label_bps=label_bps,
        depth_10bps=depth_10bps,
        visible_depth_usage=visible_depth_usage,
        conservative_net=conservative_net,
    )
    return LiquidationIntensityPaperGateRow(
        asset=asset,
        action=label.get("action", ""),
        label_status=label_status,
        trade_direction=trade_direction,
        candidate_size_usd=size_usd,
        label_bps_15m=label_bps,
        spread_bps=spread_bps,
        depth_10bps_notional=depth_10bps,
        visible_depth_usage=visible_depth_usage,
        conservative_cost_bps=conservative_cost,
        conservative_net_bps=conservative_net,
        gate_action=gate_action,
        reason=reason,
        next_step=_next_step(asset=asset, gate_action=gate_action),
    )


def _book_context(
    *,
    book: dict[str, list[list[str]]],
    contract_value: float,
    trade_direction: str,
) -> tuple[float, float]:
    if contract_value <= 0.0 or not book.get("bids") or not book.get("asks"):
        return 0.0, 0.0
    bid_levels = tuple(_parse_book_level(row, contract_value=contract_value) for row in book["bids"])
    ask_levels = tuple(_parse_book_level(row, contract_value=contract_value) for row in book["asks"])
    best_bid = bid_levels[0][0]
    best_ask = ask_levels[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid) * 10_000.0 if mid > 0.0 else 0.0
    if trade_direction == "long":
        depth = _depth_within_bps(ask_levels, mid=mid, side="ask", bps=10.0)
    else:
        depth = _depth_within_bps(bid_levels, mid=mid, side="bid", bps=10.0)
    return spread_bps, depth


def _trade_direction(label: dict[str, str]) -> str:
    signal_direction = int(float(label.get("direction") or 0.0))
    if label.get("label_status", "").startswith("reversal"):
        signal_direction *= -1
    return "long" if signal_direction > 0 else "short"


def _label_bps(label: dict[str, str]) -> float:
    if label.get("label_status", "").startswith("reversal"):
        return _float(label.get("reversal_return_15m"))
    return _float(label.get("continuation_return_15m"))


def _gate_action(
    *,
    label_status: str,
    label_bps: float,
    depth_10bps: float,
    visible_depth_usage: float,
    conservative_net: float,
) -> tuple[str, str]:
    if label_bps <= 0.0:
        return "blocked_by_label", "15m label is not positive"
    if depth_10bps <= 0.0:
        return "wait_for_depth", "visible depth is missing"
    if visible_depth_usage > 0.25:
        return "too_large_for_visible_depth", "candidate size uses too much visible 10bps depth"
    if conservative_net <= 0.0:
        return "blocked_by_cost", "fee, spread, and impact proxy consume the label"
    if label_status.endswith("pending_1h"):
        return "small_paper_probe_pending_1h", "15m label survives rough gate but 1h is not mature"
    return "small_paper_probe", "label survives rough fee, spread, and visible-depth gate"


def _next_step(*, asset: str, gate_action: str) -> str:
    if gate_action == "small_paper_probe":
        return f"paper-check {asset} liquidation intensity with fill, funding, stop, and repeat-event logs"
    if gate_action == "small_paper_probe_pending_1h":
        return f"wait for {asset} 1h label, then paper-check with fill, funding, stop, and repeat-event logs"
    if gate_action == "too_large_for_visible_depth":
        return f"retry {asset} at smaller size or find deeper venue"
    if gate_action == "blocked_by_cost":
        return f"keep {asset} as label evidence until cost or spread improves"
    return f"refresh {asset} execution context before paper-checking"


def _fetch_usdt_swap_instruments() -> dict[str, float]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    return {
        str(item["instId"]).removesuffix("-USDT-SWAP"): float(item.get("ctVal") or 0.0)
        for item in response.json().get("data", ())
        if str(item.get("instId", "")).endswith("-USDT-SWAP")
    }


def _fetch_okx_book(inst_id: str) -> dict[str, list[list[str]]]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/books",
        params={"instId": inst_id, "sz": "50"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data", ())
    return data[0] if data else {"bids": [], "asks": []}


def _parse_book_level(row: list[str], *, contract_value: float) -> tuple[float, float]:
    price = float(row[0])
    size = float(row[1])
    return price, price * size * contract_value


def _depth_within_bps(
    levels: tuple[tuple[float, float], ...],
    *,
    mid: float,
    side: str,
    bps: float,
) -> float:
    if side == "bid":
        threshold = mid * (1.0 - bps / 10_000.0)
        return sum(notional for price, notional in levels if price >= threshold)
    threshold = mid * (1.0 + bps / 10_000.0)
    return sum(notional for price, notional in levels if price <= threshold)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-path", type=Path, default=ROOT / "current_okx_liquidation_intensity_forward_labels.csv")
    parser.add_argument("--sizes-usd", nargs="+", type=float, default=list(DEFAULT_SIZES_USD))
    parser.add_argument("--fee-bps-per-fill", type=float, default=5.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_intensity_paper_gate.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_intensity_paper_gate.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_liquidation_intensity_paper_gate_rows(
        label_path=args.label_path,
        sizes_usd=tuple(args.sizes_usd),
        fee_bps_per_fill=args.fee_bps_per_fill,
    )
    write_liquidation_intensity_paper_gate_rows(rows, output_path=args.output_path)
    write_liquidation_intensity_paper_gate_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.trade_direction,
            f"size={row.candidate_size_usd:.0f}",
            row.gate_action,
            f"net={row.conservative_net_bps:.2f}bps",
        )


if __name__ == "__main__":
    main()
