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
class FollowupExecutionContextRow:
    timestamp: str
    priority: float
    asset: str
    source: str
    followup_type: str
    mark_price: float
    annualized_funding: float
    day_notional_volume: float
    open_interest_notional: float
    spread_bps: float | None
    near_depth_10bps_notional: float | None
    visible_depth_usage_1k: float | None
    action: str
    reason: str


def build_followup_execution_context_rows(
    *,
    queue_path: Path = ROOT / "current_followup_queue.csv",
    top: int = 30,
    broad_fill_risk_path: Path = ROOT.parent / "current_broad_alpha_paper_fill_risk_check.csv",
) -> tuple[FollowupExecutionContextRow, ...]:
    queue_rows = tuple(row for row in _read_rows(queue_path) if row.get("asset") != "*")[:top]
    queue_rows = _append_missing_broad_context_rows(queue_rows, broad_fill_risk_path)
    market_by_asset = _fetch_hyperliquid_market_contexts()
    observed_at = datetime.now(UTC).isoformat()
    rows = tuple(
        _build_row(
            queue_row=row,
            market_context=market_by_asset.get(row["asset"]),
            l2_metrics=_fetch_l2_metrics(row["asset"]) if row["asset"] in market_by_asset else None,
            timestamp=observed_at,
        )
        for row in queue_rows
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_followup_execution_context_csv(
    rows: tuple[FollowupExecutionContextRow, ...],
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
                "followup_type",
                "mark_price",
                "annualized_funding",
                "day_notional_volume",
                "open_interest_notional",
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
                    row.followup_type,
                    f"{row.mark_price:.12f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.day_notional_volume:.8f}",
                    f"{row.open_interest_notional:.8f}",
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


def write_followup_execution_context_md(
    rows: tuple[FollowupExecutionContextRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up Execution Context\n\n")
        handle.write(
            "This joins the follow-up queue to current Hyperliquid market context. "
            "It is a rough tradability screen, not a fill model.\n\n"
        )
        handle.write(
            "| asset | source | priority | funding ann | volume 24h | spread bps | depth 10bps USD | 1k usage | action | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.source} | "
                f"{row.priority:.4f} | "
                f"{row.annualized_funding:.6f} | "
                f"{row.day_notional_volume:.0f} | "
                f"{'' if row.spread_bps is None else f'{row.spread_bps:.4f}'} | "
                f"{'' if row.near_depth_10bps_notional is None else f'{row.near_depth_10bps_notional:.0f}'} | "
                f"{'' if row.visible_depth_usage_1k is None else f'{row.visible_depth_usage_1k:.6f}'} | "
                f"{row.action} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`tradable_context_ok` only means the current public venue context is "
            "not obviously blocking a small repeat observation. It does not cover "
            "account fees, maker queue, liquidation buffer, borrow, or operational risk.\n"
        )
    return output_path


def _build_row(
    *,
    queue_row: dict[str, str],
    market_context: dict[str, float] | None,
    l2_metrics: dict[str, float] | None,
    timestamp: str,
) -> FollowupExecutionContextRow:
    if market_context is None:
        return FollowupExecutionContextRow(
            timestamp=timestamp,
            priority=float(queue_row.get("priority") or "0"),
            asset=queue_row["asset"],
            source=queue_row.get("source", ""),
            followup_type=queue_row.get("followup_type", ""),
            mark_price=0.0,
            annualized_funding=0.0,
            day_notional_volume=0.0,
            open_interest_notional=0.0,
            spread_bps=None,
            near_depth_10bps_notional=None,
            visible_depth_usage_1k=None,
            action="not_hyperliquid",
            reason="asset is not in current Hyperliquid perp universe",
        )
    spread_bps = None if l2_metrics is None else l2_metrics["spread_bps"]
    near_depth = None if l2_metrics is None else l2_metrics["near_depth_10bps_notional"]
    visible_depth_usage = None if near_depth is None or near_depth <= 0.0 else 1_000.0 / near_depth
    action, reason = _action(
        day_notional_volume=market_context["day_notional_volume"],
        spread_bps=spread_bps,
        near_depth=near_depth,
        visible_depth_usage=visible_depth_usage,
    )
    return FollowupExecutionContextRow(
        timestamp=timestamp,
        priority=float(queue_row.get("priority") or "0"),
        asset=queue_row["asset"],
        source=queue_row.get("source", ""),
        followup_type=queue_row.get("followup_type", ""),
        mark_price=market_context["mark_price"],
        annualized_funding=market_context["annualized_funding"],
        day_notional_volume=market_context["day_notional_volume"],
        open_interest_notional=market_context["open_interest_notional"],
        spread_bps=spread_bps,
        near_depth_10bps_notional=near_depth,
        visible_depth_usage_1k=visible_depth_usage,
        action=action,
        reason=reason,
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


def _action(
    *,
    day_notional_volume: float,
    spread_bps: float | None,
    near_depth: float | None,
    visible_depth_usage: float | None,
) -> tuple[str, str]:
    if spread_bps is None or near_depth is None or visible_depth_usage is None:
        return "missing_l2_context", "could not fetch current L2 context"
    if day_notional_volume < 1_000_000.0:
        return "thin_volume_watch", "24h notional volume is low for repeat observation"
    if spread_bps > 10.0:
        return "wide_spread_watch", "current spread is wide for a small directional repeat"
    if visible_depth_usage > 0.25:
        return "thin_near_depth_watch", "1k notional uses too much visible 10 bps depth"
    return "tradable_context_ok", "public venue context does not obviously block a small repeat"


def _sort_key(row: FollowupExecutionContextRow) -> tuple[int, float, float]:
    action_priority = {
        "tradable_context_ok": 3,
        "thin_volume_watch": 2,
        "wide_spread_watch": 1,
        "thin_near_depth_watch": 1,
        "missing_l2_context": 0,
        "not_hyperliquid": -1,
    }
    return (
        action_priority.get(row.action, 0),
        row.priority,
        row.day_notional_volume,
    )


def _append_missing_broad_context_rows(
    queue_rows: tuple[dict[str, str], ...],
    broad_fill_risk_path: Path,
) -> tuple[dict[str, str], ...]:
    seen_assets = {row.get("asset", "") for row in queue_rows}
    extra_rows: list[dict[str, str]] = []
    for row in _read_rows(broad_fill_risk_path):
        asset = row.get("asset", "")
        if row.get("risk_action") != "missing_execution_context" or not asset or asset in seen_assets:
            continue
        extra_rows.append(
            {
                "priority": row.get("directional_return_bps", ""),
                "asset": asset,
                "source": "broad_alpha_paper",
                "followup_type": "missing_execution_context_backfill",
            }
        )
        seen_assets.add(asset)
    return (*queue_rows, *extra_rows)


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
        "--queue-path",
        type=Path,
        default=ROOT / "current_followup_queue.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_followup_execution_context.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_followup_execution_context.md",
    )
    parser.add_argument(
        "--broad-fill-risk-path",
        type=Path,
        default=ROOT.parent / "current_broad_alpha_paper_fill_risk_check.csv",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_followup_execution_context_rows(
        queue_path=args.queue_path,
        top=args.top,
        broad_fill_risk_path=args.broad_fill_risk_path,
    )
    write_followup_execution_context_csv(rows, output_path=args.output_path)
    write_followup_execution_context_md(rows, output_path=args.md_output_path, top=args.top)
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
