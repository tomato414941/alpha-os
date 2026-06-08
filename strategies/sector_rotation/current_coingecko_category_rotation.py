from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log10
from pathlib import Path

import requests


COINGECKO_CATEGORIES_URL = "https://api.coingecko.com/api/v3/coins/categories"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CategoryRotationRow:
    timestamp: str
    category_id: str
    name: str
    market_cap: float
    market_cap_change_24h: float
    volume_24h: float
    top_3_coins_id: str
    action: str
    score: float
    reason: str


def fetch_coingecko_categories(
    *,
    url: str = COINGECKO_CATEGORIES_URL,
) -> tuple[dict[str, object], ...]:
    try:
        response = requests.get(url, timeout=30)
        if response.status_code in {403, 429}:
            return ()
        response.raise_for_status()
    except requests.RequestException:
        return ()
    return tuple(response.json())


def build_category_rotation_rows(
    *,
    timestamp: str | None = None,
) -> tuple[CategoryRotationRow, ...]:
    observed_at = timestamp or datetime.now(UTC).isoformat()
    rows = tuple(_build_row(row=row, timestamp=observed_at) for row in fetch_coingecko_categories())
    candidates = tuple(row for row in rows if row.action != "ignore")
    return tuple(sorted(candidates, key=lambda row: row.score, reverse=True))


def write_category_rotation_csv(
    rows: tuple[CategoryRotationRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "timestamp",
                "category_id",
                "name",
                "market_cap",
                "market_cap_change_24h",
                "volume_24h",
                "top_3_coins_id",
                "action",
                "score",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.timestamp,
                    row.category_id,
                    row.name,
                    f"{row.market_cap:.4f}",
                    f"{row.market_cap_change_24h:.8f}",
                    f"{row.volume_24h:.4f}",
                    row.top_3_coins_id,
                    row.action,
                    f"{row.score:.6f}",
                    row.reason,
                )
            )
    return output_path


def write_category_rotation_md(
    rows: tuple[CategoryRotationRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current CoinGecko Category Rotation\n\n")
        handle.write(
            "This ranks public CoinGecko crypto categories by 24h market-cap "
            "change, scale, and volume. It is a sector-rotation context probe, "
            "not a trade instruction.\n\n"
        )
        handle.write(
            "| category | 24h change | market cap | volume 24h | top coins | action | score |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | --- | --- | ---: |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.name} | "
                f"{row.market_cap_change_24h:.4f} | "
                f"{row.market_cap:.0f} | "
                f"{row.volume_24h:.0f} | "
                f"{row.top_3_coins_id} | "
                f"{row.action} | "
                f"{row.score:.4f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This only shows current category rotation. It needs constituent mapping, "
            "tradable universe checks, forward labels, and liquidity/cost filters "
            "before becoming a strategy candidate.\n"
        )
    return output_path


def _build_row(*, row: dict[str, object], timestamp: str) -> CategoryRotationRow:
    market_cap = _float(row.get("market_cap"))
    change = _float(row.get("market_cap_change_24h"))
    volume = _float(row.get("volume_24h"))
    top_3 = tuple(str(value) for value in row.get("top_3_coins_id") or ())
    action, reason = _classify(change=change, market_cap=market_cap, volume=volume)
    return CategoryRotationRow(
        timestamp=timestamp,
        category_id=str(row.get("id") or ""),
        name=str(row.get("name") or ""),
        market_cap=market_cap,
        market_cap_change_24h=change,
        volume_24h=volume,
        top_3_coins_id=";".join(top_3),
        action=action,
        score=_score(change=change, market_cap=market_cap, volume=volume),
        reason=reason,
    )


def _classify(*, change: float, market_cap: float, volume: float) -> tuple[str, str]:
    if market_cap <= 0.0 or volume <= 0.0:
        return "ignore", "missing market cap or volume"
    if change >= 8.0:
        return "sector_momentum_watch", "category is showing strong positive 24h rotation"
    if change <= -8.0:
        return "sector_stress_watch", "category is showing strong negative 24h rotation"
    if abs(change) >= 4.0:
        return "sector_move_context", "category has a moderate 24h move"
    return "ignore", "category move is too small for current triage"


def _score(*, change: float, market_cap: float, volume: float) -> float:
    scale = log10(market_cap + 1.0) + log10(volume + 1.0)
    return abs(change) * scale


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
        "--output-path",
        type=Path,
        default=ROOT / "current_coingecko_category_rotation.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_coingecko_category_rotation.md",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_category_rotation_rows()
    write_category_rotation_csv(rows, output_path=args.output_path)
    write_category_rotation_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.name,
            row.action,
            f"change={row.market_cap_change_24h:.2f}",
            f"score={row.score:.2f}",
        )


if __name__ == "__main__":
    main()
