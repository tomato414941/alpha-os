from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
BINANCE_FAPI_URL = "https://fapi.binance.com"
OKX_BASE_URL = "https://www.okx.com"
FEATURE_ENDPOINTS = {
    "count_long_short_ratio": "/futures/data/globalLongShortAccountRatio",
    "count_top_long_short_ratio": "/futures/data/topLongShortAccountRatio",
    "sum_top_long_short_ratio": "/futures/data/topLongShortPositionRatio",
}


@dataclass(frozen=True)
class LiveExecutionGateRow:
    timestamp: str
    symbol: str
    feature: str
    action: str
    source_status: str
    live_feature_value: float | None
    recent_low_threshold: float
    recent_high_threshold: float
    live_condition: str
    venue: str
    instrument: str
    candidate_size_usd: float
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float
    side_depth_2bps_notional: float
    side_depth_5bps_notional: float
    side_depth_10bps_notional: float
    visible_depth_usage_5bps: float
    market_slippage_bps: float
    funding_rate_8h: float
    funding_return_1h_bps: float
    paper_cost_bps: float
    low_fee_round_trip_bps: float
    taker_round_trip_fee_bps: float
    low_fee_estimated_cost_bps: float
    taker_estimated_cost_bps: float
    paper_combined_net_1h_bps: float
    estimated_low_fee_net_1h_bps: float
    estimated_taker_net_1h_bps: float
    gate_action: str
    reason: str


def build_live_execution_gate_rows(
    *,
    candidates_path: Path,
    recent_labels_path: Path,
    candidate_sizes_usd: tuple[float, ...],
    low_fee_bps_per_fill: float,
    taker_fee_bps_per_fill: float,
) -> tuple[LiveExecutionGateRow, ...]:
    timestamp = datetime.now(UTC).isoformat()
    rows: list[LiveExecutionGateRow] = []
    for candidate in _read_rows(candidates_path):
        if candidate.get("status") != "paper_intraday_cost_supported":
            continue
        symbol = candidate.get("symbol", "")
        feature = candidate.get("feature", "")
        thresholds = _feature_thresholds(
            labels_path=recent_labels_path,
            symbol=symbol,
            feature=feature,
        )
        instrument = _okx_instrument(symbol)
        feature_value, source_status = _fetch_live_feature(symbol=symbol, feature=feature)
        book = _fetch_okx_book(instrument)
        funding = _fetch_okx_funding(instrument)
        contract_value = _fetch_okx_contract_value(instrument)
        for size_usd in candidate_sizes_usd:
            rows.append(
                _build_gate_row(
                    timestamp=timestamp,
                    candidate=candidate,
                    thresholds=thresholds,
                    live_feature_value=feature_value,
                    source_status=source_status,
                    book=book,
                    funding=funding,
                    contract_value=contract_value,
                    size_usd=size_usd,
                    low_fee_bps_per_fill=low_fee_bps_per_fill,
                    taker_fee_bps_per_fill=taker_fee_bps_per_fill,
                )
            )
    return tuple(sorted(rows, key=lambda row: (row.estimated_low_fee_net_1h_bps, -row.candidate_size_usd), reverse=True))


def write_live_execution_gate_csv(
    rows: tuple[LiveExecutionGateRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "symbol",
                "feature",
                "action",
                "source_status",
                "live_feature_value",
                "recent_low_threshold",
                "recent_high_threshold",
                "live_condition",
                "venue",
                "instrument",
                "candidate_size_usd",
                "best_bid",
                "best_ask",
                "mid_price",
                "spread_bps",
                "side_depth_2bps_notional",
                "side_depth_5bps_notional",
                "side_depth_10bps_notional",
                "visible_depth_usage_5bps",
                "market_slippage_bps",
                "funding_rate_8h",
                "funding_return_1h_bps",
                "paper_cost_bps",
                "low_fee_round_trip_bps",
                "taker_round_trip_fee_bps",
                "low_fee_estimated_cost_bps",
                "taker_estimated_cost_bps",
                "paper_combined_net_1h_bps",
                "estimated_low_fee_net_1h_bps",
                "estimated_taker_net_1h_bps",
                "gate_action",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.symbol,
                    row.feature,
                    row.action,
                    row.source_status,
                    "" if row.live_feature_value is None else f"{row.live_feature_value:.12f}",
                    f"{row.recent_low_threshold:.12f}",
                    f"{row.recent_high_threshold:.12f}",
                    row.live_condition,
                    row.venue,
                    row.instrument,
                    f"{row.candidate_size_usd:.2f}",
                    f"{row.best_bid:.12f}",
                    f"{row.best_ask:.12f}",
                    f"{row.mid_price:.12f}",
                    f"{row.spread_bps:.8f}",
                    f"{row.side_depth_2bps_notional:.8f}",
                    f"{row.side_depth_5bps_notional:.8f}",
                    f"{row.side_depth_10bps_notional:.8f}",
                    f"{row.visible_depth_usage_5bps:.8f}",
                    f"{row.market_slippage_bps:.8f}",
                    f"{row.funding_rate_8h:.12f}",
                    f"{row.funding_return_1h_bps:.8f}",
                    f"{row.paper_cost_bps:.8f}",
                    f"{row.low_fee_round_trip_bps:.8f}",
                    f"{row.taker_round_trip_fee_bps:.8f}",
                    f"{row.low_fee_estimated_cost_bps:.8f}",
                    f"{row.taker_estimated_cost_bps:.8f}",
                    f"{row.paper_combined_net_1h_bps:.8f}",
                    f"{row.estimated_low_fee_net_1h_bps:.8f}",
                    f"{row.estimated_taker_net_1h_bps:.8f}",
                    row.gate_action,
                    row.reason,
                )
            )
    return output_path


def write_live_execution_gate_md(
    rows: tuple[LiveExecutionGateRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Binance Intraday Live Execution Gate\n\n")
        handle.write(
            "This checks the current execution side of Binance-derived intraday paper labels. "
            "Binance live feature endpoints may be unavailable by region, so OKX public book and funding are used for ARB perp execution context.\n\n"
        )
        handle.write(
            "| symbol | feature | action | size | source | condition | spread | depth5 | slippage | funding1h | paper net | low-fee net | taker net | gate | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.feature} | {row.action} | "
                f"{row.candidate_size_usd:.0f} | {row.source_status} | {row.live_condition} | "
                f"{row.spread_bps:.4f} | {row.side_depth_5bps_notional:.0f} | "
                f"{row.market_slippage_bps:.4f} | {row.funding_return_1h_bps:.4f} | "
                f"{row.paper_combined_net_1h_bps:.4f} | {row.estimated_low_fee_net_1h_bps:.4f} | "
                f"{row.estimated_taker_net_1h_bps:.4f} | {row.gate_action} | {row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`low_fee_paper_probe` means the visible book does not obviously kill the low-cost paper edge. "
            "It still does not prove live alpha because the live Binance feature condition may be blocked, and maker fill probability, queue position, and stop behavior are unmodeled.\n"
        )
    return output_path


def _build_gate_row(
    *,
    timestamp: str,
    candidate: dict[str, str],
    thresholds: tuple[float, float],
    live_feature_value: float | None,
    source_status: str,
    book: dict[str, object],
    funding: dict[str, str],
    contract_value: float,
    size_usd: float,
    low_fee_bps_per_fill: float,
    taker_fee_bps_per_fill: float,
) -> LiveExecutionGateRow:
    bids, asks = _levels(book, contract_value=contract_value)
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid) * 10_000.0 if mid > 0.0 else 0.0
    side_levels = bids if _direction(candidate.get("action", "")) < 0 else asks
    market_slippage_bps = _market_slippage_bps(
        levels=side_levels,
        mid=mid,
        size_usd=size_usd,
        side="sell" if _direction(candidate.get("action", "")) < 0 else "buy",
    )
    funding_rate = float(funding.get("fundingRate") or 0.0)
    funding_return_1h_bps = (-_direction(candidate.get("action", "")) * funding_rate / 8.0) * 10_000.0
    low_fee_round_trip_bps = low_fee_bps_per_fill * 2.0
    taker_round_trip_fee_bps = taker_fee_bps_per_fill * 2.0
    low_fee_cost_bps = low_fee_round_trip_bps + spread_bps + market_slippage_bps - funding_return_1h_bps
    taker_cost_bps = taker_round_trip_fee_bps + spread_bps + market_slippage_bps - funding_return_1h_bps
    paper_cost_bps = float(candidate.get("round_trip_cost_bps") or 0.0)
    paper_net_bps = float(candidate.get("combined_net_mean_1h") or 0.0) * 10_000.0
    gross_paper_bps = paper_net_bps + paper_cost_bps
    low_fee_net_bps = gross_paper_bps - low_fee_cost_bps
    taker_net_bps = gross_paper_bps - taker_cost_bps
    low_threshold, high_threshold = thresholds
    live_condition = _live_condition(
        value=live_feature_value,
        action=candidate.get("action", ""),
        bucket=candidate.get("bucket", ""),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    gate_action, reason = _gate_action(
        source_status=source_status,
        live_condition=live_condition,
        low_fee_net_bps=low_fee_net_bps,
        taker_net_bps=taker_net_bps,
        spread_bps=spread_bps,
        size_usd=size_usd,
        depth_5bps=_depth_within_bps(
            side_levels,
            mid=mid,
            side="bid" if _direction(candidate.get("action", "")) < 0 else "ask",
            bps=5.0,
        ),
    )
    depth_5bps = _depth_within_bps(
        side_levels,
        mid=mid,
        side="bid" if _direction(candidate.get("action", "")) < 0 else "ask",
        bps=5.0,
    )
    return LiveExecutionGateRow(
        timestamp=timestamp,
        symbol=candidate.get("symbol", ""),
        feature=candidate.get("feature", ""),
        action=candidate.get("action", ""),
        source_status=source_status,
        live_feature_value=live_feature_value,
        recent_low_threshold=low_threshold,
        recent_high_threshold=high_threshold,
        live_condition=live_condition,
        venue="OKX",
        instrument=_okx_instrument(candidate.get("symbol", "")),
        candidate_size_usd=size_usd,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid,
        spread_bps=spread_bps,
        side_depth_2bps_notional=_depth_within_bps(
            side_levels,
            mid=mid,
            side="bid" if _direction(candidate.get("action", "")) < 0 else "ask",
            bps=2.0,
        ),
        side_depth_5bps_notional=depth_5bps,
        side_depth_10bps_notional=_depth_within_bps(
            side_levels,
            mid=mid,
            side="bid" if _direction(candidate.get("action", "")) < 0 else "ask",
            bps=10.0,
        ),
        visible_depth_usage_5bps=size_usd / depth_5bps if depth_5bps > 0.0 else float("inf"),
        market_slippage_bps=market_slippage_bps,
        funding_rate_8h=funding_rate,
        funding_return_1h_bps=funding_return_1h_bps,
        paper_cost_bps=paper_cost_bps,
        low_fee_round_trip_bps=low_fee_round_trip_bps,
        taker_round_trip_fee_bps=taker_round_trip_fee_bps,
        low_fee_estimated_cost_bps=low_fee_cost_bps,
        taker_estimated_cost_bps=taker_cost_bps,
        paper_combined_net_1h_bps=paper_net_bps,
        estimated_low_fee_net_1h_bps=low_fee_net_bps,
        estimated_taker_net_1h_bps=taker_net_bps,
        gate_action=gate_action,
        reason=reason,
    )


def _gate_action(
    *,
    source_status: str,
    live_condition: str,
    low_fee_net_bps: float,
    taker_net_bps: float,
    spread_bps: float,
    size_usd: float,
    depth_5bps: float,
) -> tuple[str, str]:
    if source_status != "ok":
        return "feature_source_blocked", "Binance live feature endpoint is unavailable; execution context only"
    if live_condition != "active":
        return "inactive_feature", "current feature value does not match the paper-label action"
    if low_fee_net_bps > 0.0 and spread_bps <= 2.0 and depth_5bps >= size_usd * 5.0:
        if taker_net_bps > 0.0:
            return "taker_paper_probe", "rough taker and low-fee estimates remain positive"
        return "low_fee_paper_probe", "only low-fee or maker-like execution keeps the paper edge positive"
    return "execution_blocked", "visible cost/depth check does not support the low-cost paper edge"


def _live_condition(
    *,
    value: float | None,
    action: str,
    bucket: str,
    low_threshold: float,
    high_threshold: float,
) -> str:
    if value is None:
        return "unknown"
    in_bucket = value <= low_threshold if bucket == "low" else value >= high_threshold
    if action == "short_opposite":
        return "active" if not in_bucket else "inactive"
    return "active" if in_bucket else "inactive"


def _fetch_live_feature(*, symbol: str, feature: str) -> tuple[float | None, str]:
    endpoint = FEATURE_ENDPOINTS.get(feature)
    if endpoint is None:
        return None, "unsupported_feature"
    try:
        response = requests.get(
            f"{BINANCE_FAPI_URL}{endpoint}",
            params={"symbol": symbol, "period": "5m", "limit": 1},
            timeout=15,
        )
    except requests.RequestException:
        return None, "request_failed"
    if response.status_code == 451:
        return None, "binance_region_blocked"
    if response.status_code != 200:
        return None, f"http_{response.status_code}"
    data = response.json()
    if not data:
        return None, "empty"
    try:
        return float(data[-1]["longShortRatio"]), "ok"
    except (KeyError, TypeError, ValueError):
        return None, "parse_failed"


def _fetch_okx_book(inst_id: str) -> dict[str, object]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/books",
        params={"instId": inst_id, "sz": "50"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]


def _fetch_okx_funding(inst_id: str) -> dict[str, str]:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/funding-rate",
        params={"instId": inst_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]


def _fetch_okx_contract_value(inst_id: str) -> float:
    response = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={"instType": "SWAP", "instId": inst_id},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return float(data[0].get("ctVal") or 1.0) if data else 1.0


def _levels(
    book: dict[str, object],
    *,
    contract_value: float,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    bids = tuple((float(price), float(size) * float(price) * contract_value) for price, size, *_ in book["bids"])
    asks = tuple((float(price), float(size) * float(price) * contract_value) for price, size, *_ in book["asks"])
    return bids, asks


def _feature_thresholds(*, labels_path: Path, symbol: str, feature: str) -> tuple[float, float]:
    values = sorted(float(row[feature]) for row in _read_rows(labels_path) if row.get("symbol") == symbol)
    if not values:
        return 0.0, 0.0
    return values[int(len(values) * 0.25)], values[int(len(values) * 0.75)]


def _depth_within_bps(
    levels: tuple[tuple[float, float], ...],
    *,
    mid: float,
    side: str,
    bps: float,
) -> float:
    if side == "bid":
        threshold = mid * (1.0 - (bps / 10_000.0))
        return sum(notional for price, notional in levels if price >= threshold)
    threshold = mid * (1.0 + (bps / 10_000.0))
    return sum(notional for price, notional in levels if price <= threshold)


def _market_slippage_bps(
    *,
    levels: tuple[tuple[float, float], ...],
    mid: float,
    size_usd: float,
    side: str,
) -> float:
    remaining = size_usd
    filled = 0.0
    weighted_price = 0.0
    for price, notional in levels:
        take = min(remaining, notional)
        if take <= 0.0:
            continue
        weighted_price += price * take
        filled += take
        remaining -= take
        if remaining <= 0.0:
            break
    if filled <= 0.0:
        return 10_000.0
    average_price = weighted_price / filled
    if side == "sell":
        return max(((mid - average_price) / mid) * 10_000.0, 0.0)
    return max(((average_price - mid) / mid) * 10_000.0, 0.0)


def _direction(action: str) -> int:
    if action.startswith("short"):
        return -1
    return 1


def _okx_instrument(symbol: str) -> str:
    asset = symbol.removesuffix("USDT").removesuffix("USD")
    return f"{asset}-USDT-SWAP"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_sizes(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_paper_labels_2bps.csv",
    )
    parser.add_argument(
        "--recent-labels-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_feature_labels.csv",
    )
    parser.add_argument("--candidate-sizes-usd", type=_parse_sizes, default=_parse_sizes("100,250,1000"))
    parser.add_argument("--low-fee-bps-per-fill", type=float, default=1.0)
    parser.add_argument("--taker-fee-bps-per-fill", type=float, default=5.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_live_execution_gate.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_live_execution_gate.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_live_execution_gate_rows(
        candidates_path=args.candidates_path,
        recent_labels_path=args.recent_labels_path,
        candidate_sizes_usd=args.candidate_sizes_usd,
        low_fee_bps_per_fill=args.low_fee_bps_per_fill,
        taker_fee_bps_per_fill=args.taker_fee_bps_per_fill,
    )
    write_live_execution_gate_csv(rows, output_path=args.output_path)
    write_live_execution_gate_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.feature, row.action, row.gate_action, f"low_fee_net={row.estimated_low_fee_net_1h_bps:.4f}")


if __name__ == "__main__":
    main()
