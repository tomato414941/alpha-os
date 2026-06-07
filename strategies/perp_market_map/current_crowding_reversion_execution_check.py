from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


@dataclass(frozen=True)
class CrowdingExecutionCheckRow:
    timestamp: str
    asset: str
    action: str
    status: str
    candidate_size_usd: float
    net_1h_proxy_bps: float
    funding_return_1h_bps: float
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float
    side_depth_10bps_notional: float
    side_depth_50bps_notional: float
    visible_depth_usage_10bps: float
    visible_depth_impact_bps: float
    round_trip_fee_bps: float
    conservative_cost_bps: float
    conservative_net_1h_bps: float
    gate_action: str
    reason: str


def build_crowding_execution_check_rows(
    *,
    candidate_path: Path = ROOT / "current_crowding_reversion_validated_candidates.csv",
    sizes_usd: tuple[float, ...] = (250.0, 1000.0, 2500.0),
    fee_bps_per_fill: float = 5.0,
    top: int = 10,
) -> tuple[CrowdingExecutionCheckRow, ...]:
    timestamp = datetime.now(UTC).isoformat()
    candidates = tuple(
        row
        for row in _read_rows(candidate_path)
        if row.get("status") == "paper_validated_carry_reversion_candidate"
    )[:top]
    books = {row.get("asset", ""): _fetch_l2_book(row.get("asset", "")) for row in candidates}
    output: list[CrowdingExecutionCheckRow] = []
    for candidate in candidates:
        book = books.get(candidate.get("asset", ""))
        if not book:
            continue
        for size_usd in sizes_usd:
            output.append(
                _build_row(
                    candidate=candidate,
                    book=book,
                    timestamp=timestamp,
                    size_usd=size_usd,
                    fee_bps_per_fill=fee_bps_per_fill,
                )
            )
    return tuple(sorted(output, key=_sort_key, reverse=True))


def write_crowding_execution_check_csv(
    rows: tuple[CrowdingExecutionCheckRow, ...],
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
                "action",
                "status",
                "candidate_size_usd",
                "net_1h_proxy_bps",
                "funding_return_1h_bps",
                "best_bid",
                "best_ask",
                "mid_price",
                "spread_bps",
                "side_depth_10bps_notional",
                "side_depth_50bps_notional",
                "visible_depth_usage_10bps",
                "visible_depth_impact_bps",
                "round_trip_fee_bps",
                "conservative_cost_bps",
                "conservative_net_1h_bps",
                "gate_action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.asset,
                    row.action,
                    row.status,
                    f"{row.candidate_size_usd:.2f}",
                    f"{row.net_1h_proxy_bps:.8f}",
                    f"{row.funding_return_1h_bps:.8f}",
                    f"{row.best_bid:.12f}",
                    f"{row.best_ask:.12f}",
                    f"{row.mid_price:.12f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.side_depth_10bps_notional:.8f}",
                    f"{row.side_depth_50bps_notional:.8f}",
                    f"{row.visible_depth_usage_10bps:.8f}",
                    f"{row.visible_depth_impact_bps:.8f}",
                    f"{row.round_trip_fee_bps:.8f}",
                    f"{row.conservative_cost_bps:.8f}",
                    f"{row.conservative_net_1h_bps:.8f}",
                    row.gate_action,
                    row.reason,
                )
            )
    return output_path


def write_crowding_execution_check_md(
    rows: tuple[CrowdingExecutionCheckRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tradable = tuple(row for row in rows if row.gate_action == "paper_execution_probe")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowding Reversion Execution Check\n\n")
        handle.write(
            "This applies a rough taker-fee, spread, and visible-depth gate to validated "
            "Hyperliquid carry-reversion candidates. It is still not a fill model.\n\n"
        )
        handle.write(f"- rows: `{len(rows)}`\n")
        handle.write(f"- paper execution probes: `{len(tradable)}`\n\n")
        handle.write(
            "| asset | action | size | gate | net 1h bps | cost bps | conservative bps | spread | depth10 | usage | reason |\n"
        )
        handle.write("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.asset} | {row.action} | {row.candidate_size_usd:.0f} | {row.gate_action} | "
                f"{row.net_1h_proxy_bps:.2f} | {row.conservative_cost_bps:.2f} | "
                f"{row.conservative_net_1h_bps:.2f} | {row.spread_bps:.2f} | "
                f"{row.side_depth_10bps_notional:.0f} | {row.visible_depth_usage_10bps:.4f} | "
                f"{_escape(row.reason)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_execution_probe` only means the current public book does not obviously "
            "block a small paper probe. It still excludes queue position, mark/index basis, "
            "stop behavior, funding timing, and repeated adverse selection.\n"
        )
    return output_path


def _build_row(
    *,
    candidate: dict[str, str],
    book: dict[str, object],
    timestamp: str,
    size_usd: float,
    fee_bps_per_fill: float,
) -> CrowdingExecutionCheckRow:
    bids, asks = book["levels"]
    best_bid = _price(bids[0])
    best_ask = _price(asks[0])
    mid_price = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid_price) * 10_000.0 if mid_price > 0.0 else 0.0
    direction = _direction(candidate.get("action", ""))
    side_levels = bids if direction < 0 else asks
    side_depth_10bps_notional = _notional_depth_within_bps(
        side_levels,
        mid_price=mid_price,
        bps=10.0,
        side="bid" if direction < 0 else "ask",
    )
    side_depth_50bps_notional = _notional_depth_within_bps(
        side_levels,
        mid_price=mid_price,
        bps=50.0,
        side="bid" if direction < 0 else "ask",
    )
    usage = size_usd / side_depth_10bps_notional if side_depth_10bps_notional > 0.0 else float("inf")
    visible_depth_impact_bps = min(usage, 1.0) * 10.0 if usage != float("inf") else 10.0
    round_trip_fee_bps = fee_bps_per_fill * 2.0
    conservative_cost_bps = round_trip_fee_bps + spread_bps + visible_depth_impact_bps
    net_1h_proxy_bps = _float(candidate.get("net_directional_return_1h_proxy")) * 10_000.0
    conservative_net_1h_bps = net_1h_proxy_bps - conservative_cost_bps
    gate_action, reason = _gate_action(
        conservative_net_1h_bps=conservative_net_1h_bps,
        spread_bps=spread_bps,
        usage=usage,
        side_depth_10bps_notional=side_depth_10bps_notional,
    )
    return CrowdingExecutionCheckRow(
        timestamp=timestamp,
        asset=candidate.get("asset", ""),
        action=candidate.get("action", ""),
        status=candidate.get("status", ""),
        candidate_size_usd=size_usd,
        net_1h_proxy_bps=net_1h_proxy_bps,
        funding_return_1h_bps=_float(candidate.get("funding_return_1h")) * 10_000.0,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread_bps=spread_bps,
        side_depth_10bps_notional=side_depth_10bps_notional,
        side_depth_50bps_notional=side_depth_50bps_notional,
        visible_depth_usage_10bps=usage,
        visible_depth_impact_bps=visible_depth_impact_bps,
        round_trip_fee_bps=round_trip_fee_bps,
        conservative_cost_bps=conservative_cost_bps,
        conservative_net_1h_bps=conservative_net_1h_bps,
        gate_action=gate_action,
        reason=reason,
    )


def _fetch_l2_book(asset: str, url: str = HYPERLIQUID_INFO_URL) -> dict[str, object]:
    if not asset:
        return {}
    try:
        response = requests.post(url, json={"type": "l2Book", "coin": asset}, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return {}
    payload = response.json()
    return payload if isinstance(payload, dict) and payload.get("levels") else {}


def _notional_depth_within_bps(
    levels: list[dict[str, object]],
    *,
    mid_price: float,
    bps: float,
    side: str,
) -> float:
    if side == "bid":
        threshold = mid_price * (1.0 - (bps / 10_000.0))
        return sum(_price(level) * _size(level) for level in levels if _price(level) >= threshold)
    threshold = mid_price * (1.0 + (bps / 10_000.0))
    return sum(_price(level) * _size(level) for level in levels if _price(level) <= threshold)


def _gate_action(
    *,
    conservative_net_1h_bps: float,
    spread_bps: float,
    usage: float,
    side_depth_10bps_notional: float,
) -> tuple[str, str]:
    if side_depth_10bps_notional <= 0.0:
        return "no_visible_depth", "no visible near-touch depth on the execution side"
    if usage > 0.25:
        return "too_large_for_visible_depth", "candidate size uses too much visible near-touch depth"
    if conservative_net_1h_bps <= 0.0:
        return "no_edge_after_rough_cost", "rough fees, spread, and impact consume the current proxy edge"
    if spread_bps > 15.0:
        return "wide_spread_watch", "edge survives rough cost but spread is wide"
    return "paper_execution_probe", "public book does not obviously block a small paper probe"


def _sort_key(row: CrowdingExecutionCheckRow) -> tuple[int, float, float]:
    gate_rank = {
        "paper_execution_probe": 4,
        "wide_spread_watch": 3,
        "no_edge_after_rough_cost": 2,
        "too_large_for_visible_depth": 1,
        "no_visible_depth": 0,
    }.get(row.gate_action, 0)
    return (gate_rank, row.conservative_net_1h_bps, -row.candidate_size_usd)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _price(level: dict[str, object]) -> float:
    return float(level.get("px") or "0")


def _size(level: dict[str, object]) -> float:
    return float(level.get("sz") or "0")


def _direction(action: str) -> int:
    if action.startswith("long_"):
        return 1
    if action.startswith("short_"):
        return -1
    return 0


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_validated_candidates.csv",
    )
    parser.add_argument("--sizes-usd", nargs="+", type=float, default=[250.0, 1000.0, 2500.0])
    parser.add_argument("--fee-bps-per-fill", type=float, default=5.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_execution_check.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_execution_check.md",
    )
    args = parser.parse_args()

    rows = build_crowding_execution_check_rows(
        candidate_path=args.candidate_path,
        sizes_usd=tuple(args.sizes_usd),
        fee_bps_per_fill=args.fee_bps_per_fill,
        top=args.top,
    )
    write_crowding_execution_check_csv(rows, output_path=args.output_path)
    write_crowding_execution_check_md(rows, output_path=args.md_output_path)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.gate_action,
            f"size={row.candidate_size_usd:.0f}",
            f"net={row.conservative_net_1h_bps:.2f}bps",
            f"usage={row.visible_depth_usage_10bps:.4f}",
        )


if __name__ == "__main__":
    main()
