from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class ProbeTarget:
    category: str
    name: str
    method: str
    url: str
    json_body: dict[str, object] | None = None


PROBE_TARGETS = (
    ProbeTarget(
        category="event_flow",
        name="binance_spot_aggtrades_monthly",
        method="HEAD",
        url="https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip",
    ),
    ProbeTarget(
        category="event_flow",
        name="binance_spot_trades_monthly",
        method="HEAD",
        url="https://data.binance.vision/data/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2024-01.zip",
    ),
    ProbeTarget(
        category="event_flow",
        name="binance_spot_1m_klines_monthly",
        method="HEAD",
        url="https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip",
    ),
    ProbeTarget(
        category="event_flow",
        name="binance_um_aggtrades_monthly",
        method="HEAD",
        url="https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip",
    ),
    ProbeTarget(
        category="event_flow",
        name="binance_um_trades_monthly",
        method="HEAD",
        url="https://data.binance.vision/data/futures/um/monthly/trades/BTCUSDT/BTCUSDT-trades-2024-01.zip",
    ),
    ProbeTarget(
        category="event_flow",
        name="binance_um_1m_klines_monthly",
        method="HEAD",
        url="https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip",
    ),
    ProbeTarget(
        category="lob",
        name="binance_um_book_depth_monthly_probe",
        method="HEAD",
        url="https://data.binance.vision/data/futures/um/monthly/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-01.zip",
    ),
    ProbeTarget(
        category="defi",
        name="defillama_yield_pools",
        method="GET",
        url="https://yields.llama.fi/pools",
    ),
    ProbeTarget(
        category="dex_pool_flow",
        name="geckoterminal_trending_pools",
        method="GET",
        url="https://api.geckoterminal.com/api/v2/networks/trending_pools",
    ),
    ProbeTarget(
        category="exchange",
        name="coinbase_products",
        method="GET",
        url="https://api.exchange.coinbase.com/products",
    ),
    ProbeTarget(
        category="perp_dex",
        name="hyperliquid_meta",
        method="POST",
        url="https://api.hyperliquid.xyz/info",
        json_body={"type": "meta"},
    ),
    ProbeTarget(
        category="cross_exchange",
        name="hyperliquid_predicted_fundings",
        method="POST",
        url="https://api.hyperliquid.xyz/info",
        json_body={"type": "predictedFundings"},
    ),
)


def run_probe(targets: tuple[ProbeTarget, ...] = PROBE_TARGETS) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for target in targets:
        try:
            response = _request(target)
            available = 200 <= response.status_code < 300
            rows.append(
                {
                    "category": target.category,
                    "name": target.name,
                    "method": target.method,
                    "url": target.url,
                    "status_code": response.status_code,
                    "available": available,
                    "content_length": response.headers.get("content-length", ""),
                    "notes": _notes(target, response.status_code, available),
                }
            )
        except requests.RequestException as exc:
            rows.append(
                {
                    "category": target.category,
                    "name": target.name,
                    "method": target.method,
                    "url": target.url,
                    "status_code": "",
                    "available": False,
                    "content_length": "",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
    return tuple(rows)


def write_probe_rows(rows: tuple[dict[str, object], ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "category",
                "name",
                "method",
                "url",
                "status_code",
                "available",
                "content_length",
                "notes",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def _request(target: ProbeTarget) -> requests.Response:
    if target.method == "POST":
        return requests.post(target.url, json=target.json_body, timeout=20)
    if target.method == "GET":
        return requests.get(target.url, timeout=20)
    response = requests.head(target.url, timeout=20)
    if response.status_code in (403, 405):
        return requests.get(target.url, timeout=20)
    return response


def _notes(target: ProbeTarget, status_code: int, available: bool) -> str:
    if not available:
        return "not available from this probe"
    if target.category == "event_flow":
        return "historical event-flow data path is available"
    if target.category == "defi":
        return "current DeFi yield pool data path is available"
    if target.category == "dex_pool_flow":
        return "current DEX pool-flow data path is available"
    if target.category == "perp_dex":
        return "perp DEX market metadata path is available"
    if target.category == "cross_exchange":
        return "cross-exchange predicted funding path is available"
    if target.category == "exchange":
        return "exchange product discovery path is available"
    return f"available with status {status_code}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "data_source_probe.csv",
    )
    args = parser.parse_args()

    rows = run_probe()
    write_probe_rows(rows, output_path=args.output_path)
    for row in rows:
        print(
            row["category"],
            row["name"],
            row["status_code"],
            row["available"],
            row["content_length"],
        )


if __name__ == "__main__":
    main()
