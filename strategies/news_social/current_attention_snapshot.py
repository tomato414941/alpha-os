from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


FEAR_GREED_URL = "https://api.alternative.me/fng/"
COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"


@dataclass(frozen=True)
class AttentionRow:
    timestamp: str
    source: str
    rank: int
    asset_id: str
    symbol: str
    name: str
    score: float
    label: str
    value: str


def fetch_fear_greed(url: str = FEAR_GREED_URL) -> dict[str, object]:
    response = requests.get(url, params={"limit": 30}, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_coingecko_trending(url: str = COINGECKO_TRENDING_URL) -> dict[str, object]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def build_attention_rows(
    *,
    fear_greed_payload: dict[str, object],
    trending_payload: dict[str, object],
    timestamp: str | None = None,
) -> tuple[AttentionRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    rows: list[AttentionRow] = []
    fear_greed_rows = fear_greed_payload.get("data") or ()
    for rank, item in enumerate(fear_greed_rows[:10], start=1):
        rows.append(
            AttentionRow(
                timestamp=observed_at,
                source="alternative_me_fear_greed",
                rank=rank,
                asset_id="crypto_market",
                symbol="MARKET",
                name="Crypto Fear and Greed Index",
                score=float(item["value"]),
                label=str(item["value_classification"]),
                value=str(item["timestamp"]),
            )
        )
    for rank, item in enumerate(trending_payload.get("coins") or (), start=1):
        coin = item["item"]
        rows.append(
            AttentionRow(
                timestamp=observed_at,
                source="coingecko_trending",
                rank=rank,
                asset_id=str(coin.get("id") or ""),
                symbol=str(coin.get("symbol") or ""),
                name=str(coin.get("name") or ""),
                score=float(coin.get("score") or rank),
                label=str(coin.get("market_cap_rank") or ""),
                value=str(_usd_price_change_24h(coin)),
            )
        )
    return tuple(rows)


def write_attention_rows(rows: tuple[AttentionRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "source",
                "rank",
                "asset_id",
                "symbol",
                "name",
                "score",
                "label",
                "value",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.source,
                    row.rank,
                    row.asset_id,
                    row.symbol,
                    row.name,
                    f"{row.score:.8f}",
                    row.label,
                    row.value,
                )
            )
    return output_path


def _usd_price_change_24h(coin: dict[str, object]) -> float:
    data = coin.get("data") or {}
    changes = data.get("price_change_percentage_24h") if isinstance(data, dict) else {}
    if not isinstance(changes, dict):
        return 0.0
    return float(changes.get("usd") or changes.get("bmd") or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "current_attention_snapshot.csv",
    )
    args = parser.parse_args()

    rows = build_attention_rows(
        fear_greed_payload=fetch_fear_greed(),
        trending_payload=fetch_coingecko_trending(),
    )
    write_attention_rows(rows, output_path=args.output_path)
    for row in rows[:20]:
        print(row.source, row.rank, row.symbol, row.name, row.score, row.label, row.value)


if __name__ == "__main__":
    main()

