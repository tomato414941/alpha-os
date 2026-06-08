from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VolumePriceDislocationExecutionGateRow:
    timestamp: str
    observed_at: str
    symbol: str
    name: str
    side: str
    direction: int
    score: float
    label_status: str
    directional_return_1h: float | None
    directional_return_4h: float | None
    price_source: str
    mark_price: float
    annualized_funding: float
    day_notional_volume: float
    open_interest_notional: float
    spread_bps: float | None
    near_depth_10bps_notional: float | None
    visible_depth_usage_250: float | None
    conservative_net_4h_bps: float | None
    action: str
    reason: str
    next_step: str


def build_volume_price_dislocation_execution_gate_rows(
    *,
    labels_path: Path = ROOT / "current_volume_price_dislocation_labels.csv",
    top: int = 40,
    target_notional: float = 250.0,
    taker_fee_bps_per_fill: float = 4.0,
) -> tuple[VolumePriceDislocationExecutionGateRow, ...]:
    labels = _supported_or_labeled_rows(_read_rows(labels_path))[:top]
    market_by_asset = _fetch_hyperliquid_market_contexts()
    timestamp = datetime.now(UTC).isoformat()
    rows = tuple(
        _build_row(
            label=row,
            market_context=market_by_asset.get(row.get("symbol", "")),
            l2_metrics=_fetch_l2_metrics(row.get("symbol", ""))
            if row.get("symbol", "") in market_by_asset
            else None,
            timestamp=timestamp,
            target_notional=target_notional,
            taker_fee_bps_per_fill=taker_fee_bps_per_fill,
        )
        for row in labels
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_volume_price_dislocation_execution_gate_csv(
    rows: tuple[VolumePriceDislocationExecutionGateRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "observed_at",
                "symbol",
                "name",
                "side",
                "direction",
                "score",
                "label_status",
                "directional_return_1h",
                "directional_return_4h",
                "price_source",
                "mark_price",
                "annualized_funding",
                "day_notional_volume",
                "open_interest_notional",
                "spread_bps",
                "near_depth_10bps_notional",
                "visible_depth_usage_250",
                "conservative_net_4h_bps",
                "action",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.observed_at,
                    row.symbol,
                    row.name,
                    row.side,
                    row.direction,
                    f"{row.score:.8f}",
                    row.label_status,
                    _format_optional(row.directional_return_1h),
                    _format_optional(row.directional_return_4h),
                    row.price_source,
                    f"{row.mark_price:.12f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.day_notional_volume:.8f}",
                    f"{row.open_interest_notional:.8f}",
                    _format_optional(row.spread_bps),
                    _format_optional(row.near_depth_10bps_notional),
                    _format_optional(row.visible_depth_usage_250),
                    _format_optional(row.conservative_net_4h_bps),
                    row.action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_volume_price_dislocation_execution_gate_md(
    rows: tuple[VolumePriceDislocationExecutionGateRow, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Volume Price Dislocation Execution Gate\n\n")
        handle.write(
            "This joins supported volume-price dislocation labels to current Hyperliquid "
            "funding, spread, and public book depth. It is a rough paper gate, not a fill model.\n\n"
        )
        handle.write(
            "| symbol | side | dir 1h | dir 4h | funding ann | spread bps | depth 10bps USD | "
            "250 usage | net 4h bps | action | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.symbol} | "
                f"{row.side} | "
                f"{_format_optional(row.directional_return_1h)} | "
                f"{_format_optional(row.directional_return_4h)} | "
                f"{row.annualized_funding:.6f} | "
                f"{_format_optional(row.spread_bps)} | "
                f"{_format_optional(row.near_depth_10bps_notional)} | "
                f"{_format_optional(row.visible_depth_usage_250)} | "
                f"{_format_optional(row.conservative_net_4h_bps)} | "
                f"{row.action} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_execution_probe` only means the current public venue context does not "
            "obviously kill a small repeat observation. It excludes queue position, account "
            "fees, realized fills, stop behavior, and whether the 4h label repeats.\n"
        )
    return output_path


def _supported_or_labeled_rows(rows: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
    candidates = tuple(
        row
        for row in rows
        if row.get("directional_return_1h", "") != "" or row.get("directional_return_4h", "") != ""
    )
    return tuple(
        sorted(
            candidates,
            key=lambda row: (
                _float(row.get("directional_return_4h")),
                _float(row.get("directional_return_1h")),
                _float(row.get("score")),
            ),
            reverse=True,
        )
    )


def _build_row(
    *,
    label: dict[str, str],
    market_context: dict[str, float] | None,
    l2_metrics: dict[str, float] | None,
    timestamp: str,
    target_notional: float,
    taker_fee_bps_per_fill: float,
) -> VolumePriceDislocationExecutionGateRow:
    symbol = label.get("symbol", "")
    direction = int(float(label.get("direction") or "0"))
    dir_4h = _optional_float(label.get("directional_return_4h"))
    if market_context is None:
        return _row(
            label=label,
            timestamp=timestamp,
            action="not_hyperliquid",
            reason="symbol is not in current Hyperliquid perp universe",
            next_step=f"skip {symbol} market-breadth paper probe unless another executable venue is added",
        )
    spread_bps = None if l2_metrics is None else l2_metrics["spread_bps"]
    near_depth = None if l2_metrics is None else l2_metrics["near_depth_10bps_notional"]
    visible_depth_usage = None if near_depth is None or near_depth <= 0.0 else target_notional / near_depth
    conservative_net_4h_bps = _conservative_net_4h_bps(
        directional_return_4h=dir_4h,
        direction=direction,
        annualized_funding=market_context["annualized_funding"],
        spread_bps=spread_bps,
        taker_fee_bps_per_fill=taker_fee_bps_per_fill,
    )
    action, reason = _action(
        directional_return_4h=dir_4h,
        conservative_net_4h_bps=conservative_net_4h_bps,
        day_notional_volume=market_context["day_notional_volume"],
        spread_bps=spread_bps,
        visible_depth_usage=visible_depth_usage,
    )
    return _row(
        label=label,
        timestamp=timestamp,
        mark_price=market_context["mark_price"],
        annualized_funding=market_context["annualized_funding"],
        day_notional_volume=market_context["day_notional_volume"],
        open_interest_notional=market_context["open_interest_notional"],
        spread_bps=spread_bps,
        near_depth_10bps_notional=near_depth,
        visible_depth_usage_250=visible_depth_usage,
        conservative_net_4h_bps=conservative_net_4h_bps,
        action=action,
        reason=reason,
        next_step=_next_step(symbol=symbol, action=action),
    )


def _row(
    *,
    label: dict[str, str],
    timestamp: str,
    action: str,
    reason: str,
    next_step: str,
    mark_price: float = 0.0,
    annualized_funding: float = 0.0,
    day_notional_volume: float = 0.0,
    open_interest_notional: float = 0.0,
    spread_bps: float | None = None,
    near_depth_10bps_notional: float | None = None,
    visible_depth_usage_250: float | None = None,
    conservative_net_4h_bps: float | None = None,
) -> VolumePriceDislocationExecutionGateRow:
    return VolumePriceDislocationExecutionGateRow(
        timestamp=timestamp,
        observed_at=label.get("observed_at", ""),
        symbol=label.get("symbol", ""),
        name=label.get("name", ""),
        side=label.get("side", ""),
        direction=int(float(label.get("direction") or "0")),
        score=_float(label.get("score")),
        label_status=label.get("label_status", ""),
        directional_return_1h=_optional_float(label.get("directional_return_1h")),
        directional_return_4h=_optional_float(label.get("directional_return_4h")),
        price_source=label.get("price_source", ""),
        mark_price=mark_price,
        annualized_funding=annualized_funding,
        day_notional_volume=day_notional_volume,
        open_interest_notional=open_interest_notional,
        spread_bps=spread_bps,
        near_depth_10bps_notional=near_depth_10bps_notional,
        visible_depth_usage_250=visible_depth_usage_250,
        conservative_net_4h_bps=conservative_net_4h_bps,
        action=action,
        reason=reason,
        next_step=next_step,
    )


def _fetch_hyperliquid_market_contexts() -> dict[str, dict[str, float]]:
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={"type": "metaAndAssetCtxs"},
        timeout=30,
    )
    response.raise_for_status()
    meta, contexts = response.json()
    rows: dict[str, dict[str, float]] = {}
    for asset_meta, context in zip(meta["universe"], contexts, strict=False):
        asset = str(asset_meta["name"])
        mark_price = _float(context.get("markPx"))
        rows[asset] = {
            "mark_price": mark_price,
            "annualized_funding": _float(context.get("funding")) * 24.0 * 365.0,
            "day_notional_volume": _float(context.get("dayNtlVlm")),
            "open_interest_notional": _float(context.get("openInterest")) * mark_price,
        }
    return rows


def _fetch_l2_metrics(asset: str) -> dict[str, float] | None:
    try:
        response = requests.post(
            HYPERLIQUID_INFO_URL,
            json={"type": "l2Book", "coin": asset},
            timeout=30,
        )
        response.raise_for_status()
        bids, asks = response.json()["levels"]
        best_bid = float(bids[0]["px"])
        best_ask = float(asks[0]["px"])
        mid_price = (best_bid + best_ask) / 2.0
        bid_depth = _depth_within_bps(bids, mid_price=mid_price, bps=10.0, side="bid")
        ask_depth = _depth_within_bps(asks, mid_price=mid_price, bps=10.0, side="ask")
        return {
            "spread_bps": ((best_ask - best_bid) / mid_price) * 10_000.0,
            "near_depth_10bps_notional": min(bid_depth, ask_depth) * mid_price,
        }
    except (KeyError, IndexError, TypeError, ValueError, requests.RequestException):
        return None


def _depth_within_bps(
    levels: list[dict[str, object]],
    *,
    mid_price: float,
    bps: float,
    side: str,
) -> float:
    if side == "bid":
        threshold = mid_price * (1.0 - (bps / 10_000.0))
        return sum(float(level["sz"]) for level in levels if float(level["px"]) >= threshold)
    threshold = mid_price * (1.0 + (bps / 10_000.0))
    return sum(float(level["sz"]) for level in levels if float(level["px"]) <= threshold)


def _conservative_net_4h_bps(
    *,
    directional_return_4h: float | None,
    direction: int,
    annualized_funding: float,
    spread_bps: float | None,
    taker_fee_bps_per_fill: float,
) -> float | None:
    if directional_return_4h is None or spread_bps is None:
        return None
    funding_pnl_4h = -direction * annualized_funding / (365.0 * 6.0)
    fee_bps = taker_fee_bps_per_fill * 2.0
    return (directional_return_4h + funding_pnl_4h) * 10_000.0 - fee_bps - spread_bps


def _action(
    *,
    directional_return_4h: float | None,
    conservative_net_4h_bps: float | None,
    day_notional_volume: float,
    spread_bps: float | None,
    visible_depth_usage: float | None,
) -> tuple[str, str]:
    if directional_return_4h is None:
        return "pending_4h_label", "4h label is not available yet"
    if directional_return_4h <= 0.0:
        return "label_contradicted", "4h label does not support the market-breadth direction"
    if spread_bps is None or visible_depth_usage is None or conservative_net_4h_bps is None:
        return "missing_l2_context", "could not fetch current L2 context"
    if conservative_net_4h_bps <= 0.0:
        return "no_edge_after_rough_cost", "4h label is erased by rough funding, spread, and taker-fee assumptions"
    if day_notional_volume < 1_000_000.0:
        return "thin_volume_watch", "24h notional volume is low for repeat observation"
    if spread_bps > 10.0:
        return "wide_spread_watch", "current spread is wide for a small directional repeat"
    if visible_depth_usage > 0.25:
        return "too_large_for_visible_depth", "250 USD uses too much visible 10 bps depth"
    return "paper_execution_probe", "current public venue context does not obviously block a small repeat"


def _next_step(*, symbol: str, action: str) -> str:
    if action == "paper_execution_probe":
        return f"run a small paper probe for {symbol} and log 15m/1h/4h outcome, fill, funding, and stop behavior"
    if action == "label_contradicted":
        return f"do not promote {symbol} market-breadth setup without a fresh positive 4h label"
    if action == "no_edge_after_rough_cost":
        return f"do not promote {symbol} unless fresh labels survive funding, spread, and fee assumptions"
    if action in {"wide_spread_watch", "thin_volume_watch", "too_large_for_visible_depth"}:
        return f"wait for better {symbol} venue context before paper probing this setup"
    return f"collect more executable venue context for {symbol}"


def _sort_key(row: VolumePriceDislocationExecutionGateRow) -> tuple[int, float, float]:
    action_priority = {
        "paper_execution_probe": 5,
        "thin_volume_watch": 3,
        "wide_spread_watch": 2,
        "too_large_for_visible_depth": 2,
        "no_edge_after_rough_cost": 1,
        "label_contradicted": 0,
        "pending_4h_label": 0,
        "missing_l2_context": -1,
        "not_hyperliquid": -2,
    }
    return (
        action_priority.get(row.action, 0),
        row.conservative_net_4h_bps or -1_000_000.0,
        row.score,
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _optional_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return _float(value)


def _float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-path", type=Path, default=ROOT / "current_volume_price_dislocation_labels.csv")
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_volume_price_dislocation_execution_gate.csv")
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_volume_price_dislocation_execution_gate.md",
    )
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--target-notional", type=float, default=250.0)
    parser.add_argument("--taker-fee-bps-per-fill", type=float, default=4.0)
    args = parser.parse_args()

    rows = build_volume_price_dislocation_execution_gate_rows(
        labels_path=args.labels_path,
        top=args.top,
        target_notional=args.target_notional,
        taker_fee_bps_per_fill=args.taker_fee_bps_per_fill,
    )
    write_volume_price_dislocation_execution_gate_csv(rows, output_path=args.output_path)
    write_volume_price_dislocation_execution_gate_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.action,
            row.symbol,
            f"net4h={_format_optional(row.conservative_net_4h_bps)}",
            f"spread={_format_optional(row.spread_bps)}",
        )


if __name__ == "__main__":
    main()
