from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReachabilityCheck:
    lane: str
    source: str
    method: str
    url: str
    payload: dict[str, object] | None
    history_kind: str
    access: str
    expected_use: str


@dataclass(frozen=True)
class ReachabilityResult:
    check: ReachabilityCheck
    status_code: int | None
    available: bool
    response_bytes: int
    notes: str


CHECKS = (
    ReachabilityCheck(
        lane="liquidation_oi_funding",
        source="hyperliquid_meta_and_asset_contexts",
        method="POST",
        url="https://api.hyperliquid.xyz/info",
        payload={"type": "metaAndAssetCtxs"},
        history_kind="current_snapshot",
        access="free_public",
        expected_use="current OI, funding, volume, premium, and impact context",
    ),
    ReachabilityCheck(
        lane="liquidation_oi_funding",
        source="binance_um_daily_metrics",
        method="HEAD",
        url=(
            "https://data.binance.vision/data/futures/um/daily/metrics/"
            "BTCUSDT/BTCUSDT-metrics-2024-01-01.zip"
        ),
        payload=None,
        history_kind="daily_historical_file",
        access="free_public",
        expected_use="historical OI and derivatives metrics if schema is usable",
    ),
    ReachabilityCheck(
        lane="funding_basis",
        source="binance_um_daily_premium_index_klines",
        method="HEAD",
        url=(
            "https://data.binance.vision/data/futures/um/daily/premiumIndexKlines/"
            "BTCUSDT/1m/BTCUSDT-1m-2024-01-01.zip"
        ),
        payload=None,
        history_kind="minute_historical_file",
        access="free_public",
        expected_use="premium/index/funding-adjacent history for basis tests",
    ),
    ReachabilityCheck(
        lane="funding_basis",
        source="hyperliquid_predicted_fundings",
        method="POST",
        url="https://api.hyperliquid.xyz/info",
        payload={"type": "predictedFundings"},
        history_kind="current_snapshot",
        access="free_public",
        expected_use="current multi-venue predicted funding spread",
    ),
    ReachabilityCheck(
        lane="l2_fill",
        source="hyperliquid_l2_book",
        method="POST",
        url="https://api.hyperliquid.xyz/info",
        payload={"type": "l2Book", "coin": "BTC"},
        history_kind="current_snapshot",
        access="free_public",
        expected_use="top 20 L2 levels per side for fill/adverse-selection probes",
    ),
    ReachabilityCheck(
        lane="l2_fill",
        source="hyperliquid_recent_trades",
        method="POST",
        url="https://api.hyperliquid.xyz/info",
        payload={"type": "recentTrades", "coin": "BTC"},
        history_kind="recent_snapshot",
        access="free_public",
        expected_use="recent trades to pair with L2 snapshots",
    ),
    ReachabilityCheck(
        lane="attention_liquidity",
        source="defillama_stablecoins",
        method="GET",
        url="https://stablecoins.llama.fi/stablecoins?includePrices=true",
        payload=None,
        history_kind="current_with_prev_period_fields",
        access="free_public",
        expected_use="stablecoin supply and peg context",
    ),
    ReachabilityCheck(
        lane="attention_liquidity",
        source="alternative_me_fear_greed",
        method="GET",
        url="https://api.alternative.me/fng/?limit=30",
        payload=None,
        history_kind="short_history",
        access="free_public",
        expected_use="market-level sentiment context",
    ),
    ReachabilityCheck(
        lane="attention_liquidity",
        source="coingecko_trending",
        method="GET",
        url="https://api.coingecko.com/api/v3/search/trending",
        payload=None,
        history_kind="current_snapshot",
        access="free_public",
        expected_use="attention proxy for trending assets",
    ),
)


def run_reachability_checks(checks: tuple[ReachabilityCheck, ...] = CHECKS) -> tuple[ReachabilityResult, ...]:
    return tuple(_run_check(check) for check in checks)


def write_reachability_results(
    results: tuple[ReachabilityResult, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "lane",
                "source",
                "method",
                "url",
                "status_code",
                "available",
                "response_bytes",
                "history_kind",
                "access",
                "expected_use",
                "notes",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.check.lane,
                    result.check.source,
                    result.check.method,
                    result.check.url,
                    "" if result.status_code is None else result.status_code,
                    result.available,
                    result.response_bytes,
                    result.check.history_kind,
                    result.check.access,
                    result.check.expected_use,
                    result.notes,
                )
            )
    return output_path


def _run_check(check: ReachabilityCheck) -> ReachabilityResult:
    try:
        if check.method == "POST":
            response = requests.post(check.url, json=check.payload, timeout=30)
        elif check.method == "HEAD":
            response = requests.head(check.url, timeout=30)
        else:
            response = requests.get(check.url, timeout=30)
    except requests.RequestException as exc:
        return ReachabilityResult(
            check=check,
            status_code=None,
            available=False,
            response_bytes=0,
            notes=f"request failed: {type(exc).__name__}",
        )
    response_bytes = int(response.headers.get("content-length") or len(response.content or b""))
    return ReachabilityResult(
        check=check,
        status_code=response.status_code,
        available=200 <= response.status_code < 300,
        response_bytes=response_bytes,
        notes=_notes(check, response.status_code),
    )


def _notes(check: ReachabilityCheck, status_code: int) -> str:
    if 200 <= status_code < 300:
        if check.history_kind == "current_snapshot":
            return "reachable but not historical"
        return "reachable"
    if status_code in {401, 403}:
        return "auth or permission required"
    if status_code == 451:
        return "location restricted"
    if status_code == 404:
        return "not found at this route"
    return "unexpected status"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "data_reachability.csv",
    )
    args = parser.parse_args()

    results = run_reachability_checks()
    write_reachability_results(results, output_path=args.output_path)
    for result in results:
        print(
            result.check.lane,
            result.check.source,
            result.status_code,
            result.available,
            result.notes,
        )


if __name__ == "__main__":
    main()

