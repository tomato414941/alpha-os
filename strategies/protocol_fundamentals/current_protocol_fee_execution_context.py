from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"
BINANCE_FAPI_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProtocolFeeExecutionContextRow:
    timestamp: str
    token_symbol: str
    protocol: str
    thesis_status: str
    thesis_score: float
    side: str
    fee_growth_7d: float
    price_change_7d: float
    funding: float
    hyperliquid_perp: bool
    okx_usdt_swap: bool
    binance_usdt_perp: bool
    venue_count: int
    hl_mark_price: float
    hl_annualized_funding: float
    hl_day_notional_volume: float
    hl_open_interest_notional: float
    hl_spread_bps: float | None
    hl_near_depth_10bps_notional: float | None
    hl_visible_depth_usage_1k: float | None
    action: str
    reason: str
    next_step: str


def build_protocol_fee_execution_context_rows(
    *,
    context_path: Path = ROOT / "current_protocol_fee_price_context.csv",
) -> tuple[ProtocolFeeExecutionContextRow, ...]:
    candidates = tuple(
        row
        for row in _read_rows(context_path)
        if row.get("status") == "fee_growth_price_lag_candidate"
    )
    hyperliquid_markets = _fetch_hyperliquid_market_contexts()
    okx_assets = _fetch_okx_usdt_swap_assets()
    binance_assets = _fetch_binance_usdt_perp_assets()
    observed_at = datetime.now(UTC).isoformat()
    rows = tuple(
        _build_row(
            row=row,
            observed_at=observed_at,
            hyperliquid_markets=hyperliquid_markets,
            okx_assets=okx_assets,
            binance_assets=binance_assets,
        )
        for row in candidates
    )
    return tuple(sorted(rows, key=_sort_key, reverse=True))


def write_protocol_fee_execution_context_csv(
    rows: tuple[ProtocolFeeExecutionContextRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "token_symbol",
                "protocol",
                "thesis_status",
                "thesis_score",
                "side",
                "fee_growth_7d",
                "price_change_7d",
                "funding",
                "hyperliquid_perp",
                "okx_usdt_swap",
                "binance_usdt_perp",
                "venue_count",
                "hl_mark_price",
                "hl_annualized_funding",
                "hl_day_notional_volume",
                "hl_open_interest_notional",
                "hl_spread_bps",
                "hl_near_depth_10bps_notional",
                "hl_visible_depth_usage_1k",
                "action",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.token_symbol,
                    row.protocol,
                    row.thesis_status,
                    f"{row.thesis_score:.8f}",
                    row.side,
                    f"{row.fee_growth_7d:.8f}",
                    f"{row.price_change_7d:.8f}",
                    f"{row.funding:.8f}",
                    row.hyperliquid_perp,
                    row.okx_usdt_swap,
                    row.binance_usdt_perp,
                    row.venue_count,
                    f"{row.hl_mark_price:.12f}",
                    f"{row.hl_annualized_funding:.8f}",
                    f"{row.hl_day_notional_volume:.8f}",
                    f"{row.hl_open_interest_notional:.8f}",
                    "" if row.hl_spread_bps is None else f"{row.hl_spread_bps:.8f}",
                    (
                        ""
                        if row.hl_near_depth_10bps_notional is None
                        else f"{row.hl_near_depth_10bps_notional:.8f}"
                    ),
                    "" if row.hl_visible_depth_usage_1k is None else f"{row.hl_visible_depth_usage_1k:.8f}",
                    row.action,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_protocol_fee_execution_context_md(
    rows: tuple[ProtocolFeeExecutionContextRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Protocol Fee Execution Context\n\n")
        handle.write(
            "This joins protocol fee-growth lag candidates to current perp venue "
            "coverage and Hyperliquid public-book context. It is a paper-observation "
            "gate, not a live trade instruction.\n\n"
        )
        handle.write(
            "| token | protocol | score | price7d | venues | HL funding | HL volume 24h | spread bps | depth 10bps USD | action | next step |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.token_symbol} | "
                f"{row.protocol} | "
                f"{row.thesis_score:.4f} | "
                f"{row.price_change_7d:.2f} | "
                f"{row.venue_count} | "
                f"{row.hl_annualized_funding:.4f} | "
                f"{row.hl_day_notional_volume:.0f} | "
                f"{'' if row.hl_spread_bps is None else f'{row.hl_spread_bps:.4f}'} | "
                f"{'' if row.hl_near_depth_10bps_notional is None else f'{row.hl_near_depth_10bps_notional:.0f}'} | "
                f"{row.action} | "
                f"{row.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_observation_ready` only means the fee-growth lag thesis has "
            "current venue coverage and public-book context that is not obviously "
            "blocking a small paper observation. It does not prove alpha, fill "
            "quality, borrow availability, account fees, or liquidation safety.\n"
        )
    return output_path


def _build_row(
    *,
    row: dict[str, str],
    observed_at: str,
    hyperliquid_markets: dict[str, dict[str, float]],
    okx_assets: frozenset[str],
    binance_assets: frozenset[str],
) -> ProtocolFeeExecutionContextRow:
    token = row.get("token_symbol", "")
    market_context = hyperliquid_markets.get(token)
    hl = market_context is not None
    okx = token in okx_assets
    binance = token in binance_assets
    l2_metrics = _fetch_l2_metrics(token) if hl else None
    spread_bps = None if l2_metrics is None else l2_metrics["spread_bps"]
    near_depth = None if l2_metrics is None else l2_metrics["near_depth_10bps_notional"]
    visible_depth_usage = None if near_depth is None or near_depth <= 0.0 else 1_000.0 / near_depth
    venue_count = sum((hl, okx, binance))
    action, reason, next_step = _action_reason_next_step(
        token=token,
        venue_count=venue_count,
        hyperliquid_perp=hl,
        day_notional_volume=0.0 if market_context is None else market_context["day_notional_volume"],
        spread_bps=spread_bps,
        visible_depth_usage=visible_depth_usage,
    )
    return ProtocolFeeExecutionContextRow(
        timestamp=observed_at,
        token_symbol=token,
        protocol=row.get("protocol", ""),
        thesis_status=row.get("status", ""),
        thesis_score=_float(row.get("score")),
        side=row.get("side", ""),
        fee_growth_7d=_float(row.get("fee_growth_7d")),
        price_change_7d=_float(row.get("price_change_7d")),
        funding=_float(row.get("funding")),
        hyperliquid_perp=hl,
        okx_usdt_swap=okx,
        binance_usdt_perp=binance,
        venue_count=venue_count,
        hl_mark_price=0.0 if market_context is None else market_context["mark_price"],
        hl_annualized_funding=0.0 if market_context is None else market_context["annualized_funding"],
        hl_day_notional_volume=0.0 if market_context is None else market_context["day_notional_volume"],
        hl_open_interest_notional=0.0 if market_context is None else market_context["open_interest_notional"],
        hl_spread_bps=spread_bps,
        hl_near_depth_10bps_notional=near_depth,
        hl_visible_depth_usage_1k=visible_depth_usage,
        action=action,
        reason=reason,
        next_step=next_step,
    )


def _action_reason_next_step(
    *,
    token: str,
    venue_count: int,
    hyperliquid_perp: bool,
    day_notional_volume: float,
    spread_bps: float | None,
    visible_depth_usage: float | None,
) -> tuple[str, str, str]:
    if venue_count == 0:
        return (
            "venue_gap",
            "candidate is not listed on the checked major perp venues",
            f"treat {token} as research-only until a tradable venue route is identified",
        )
    if not hyperliquid_perp:
        return (
            "non_hyperliquid_route_check",
            "candidate has venue coverage but no Hyperliquid public-book context here",
            f"check {token} OKX/Binance book depth, fees, and funding before paper observation",
        )
    if spread_bps is None or visible_depth_usage is None:
        return (
            "missing_hl_book_context",
            "Hyperliquid listing exists but current L2 context was unavailable",
            f"rerun {token} Hyperliquid L2 context before paper observation",
        )
    if day_notional_volume < 2_000_000.0:
        return (
            "thin_volume_watch",
            "Hyperliquid 24h notional volume is low for the fee-growth lag thesis",
            f"keep {token} as a low-liquidity paper label, not an execution candidate",
        )
    if spread_bps > 8.0:
        return (
            "wide_spread_watch",
            "current Hyperliquid spread is wide for a small directional paper observation",
            f"wait for tighter {token} spread or use another venue before paper observation",
        )
    if visible_depth_usage > 0.20:
        return (
            "thin_depth_watch",
            "1k notional uses too much visible 10 bps Hyperliquid depth",
            f"downsize {token} paper observation or use another venue route",
        )
    return (
        "paper_observation_ready",
        "current venue and public-book context do not obviously block a small paper observation",
        f"paper-label {token} fee-growth lag with 4h/12h/24h return, funding, spread, and depth costs",
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
        asset = str(asset_meta["name"]).upper()
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


def _fetch_okx_usdt_swap_assets() -> frozenset[str]:
    response = requests.get(
        OKX_INSTRUMENTS_URL,
        params={"instType": "SWAP"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", ())
    return frozenset(
        str(row.get("uly") or "").split("-")[0].upper()
        for row in rows
        if str(row.get("instId") or "").endswith("-USDT-SWAP")
    )


def _fetch_binance_usdt_perp_assets() -> frozenset[str]:
    try:
        response = requests.get(BINANCE_FAPI_EXCHANGE_INFO_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return frozenset()
    rows = response.json().get("symbols", ())
    return frozenset(
        str(row.get("baseAsset") or "").upper()
        for row in rows
        if row.get("contractType") == "PERPETUAL"
        and row.get("quoteAsset") == "USDT"
        and row.get("status") == "TRADING"
    )


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


def _sort_key(row: ProtocolFeeExecutionContextRow) -> tuple[int, int, float, float]:
    action_rank = {
        "paper_observation_ready": 5,
        "non_hyperliquid_route_check": 4,
        "thin_depth_watch": 3,
        "wide_spread_watch": 2,
        "thin_volume_watch": 1,
        "missing_hl_book_context": 1,
        "venue_gap": 0,
    }.get(row.action, 0)
    return (action_rank, row.venue_count, row.thesis_score, -abs(row.price_change_7d))


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value) if value not in {None, ""} else 0.0
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-path",
        type=Path,
        default=ROOT / "current_protocol_fee_price_context.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_execution_context.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_protocol_fee_execution_context.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_protocol_fee_execution_context_rows(context_path=args.context_path)
    write_protocol_fee_execution_context_csv(rows, output_path=args.output_path)
    write_protocol_fee_execution_context_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.token_symbol,
            row.action,
            f"venues={row.venue_count}",
            f"spread={'' if row.hl_spread_bps is None else f'{row.hl_spread_bps:.4f}'}",
            f"depth={'' if row.hl_near_depth_10bps_notional is None else f'{row.hl_near_depth_10bps_notional:.0f}'}",
        )


if __name__ == "__main__":
    main()
