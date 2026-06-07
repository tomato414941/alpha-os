from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.stablecoin_liquidity.current_supply_snapshot import (
    StablecoinSupplyRow,
    build_stablecoin_supply_rows,
    fetch_stablecoins,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StablecoinPegStressRow:
    symbol: str
    name: str
    peg_type: str
    peg_mechanism: str
    current_supply_usd: float
    week_change_usd: float
    month_change_usd: float
    price: float
    peg_deviation: float
    supply_stress_score: float
    score: float
    status: str
    side: str
    reason: str
    next_step: str


def build_peg_stress_rows(
    rows: tuple[StablecoinSupplyRow, ...],
    *,
    min_supply_usd: float = 50_000_000.0,
) -> tuple[StablecoinPegStressRow, ...]:
    output = tuple(_build_row(row) for row in rows if row.current_supply_usd >= min_supply_usd and row.price > 0.0)
    return tuple(sorted(output, key=lambda row: row.score, reverse=True))


def write_peg_stress_csv(rows: tuple[StablecoinPegStressRow, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "name",
                "peg_type",
                "peg_mechanism",
                "current_supply_usd",
                "week_change_usd",
                "month_change_usd",
                "price",
                "peg_deviation",
                "supply_stress_score",
                "score",
                "status",
                "side",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.name,
                    row.peg_type,
                    row.peg_mechanism,
                    f"{row.current_supply_usd:.2f}",
                    f"{row.week_change_usd:.2f}",
                    f"{row.month_change_usd:.2f}",
                    f"{row.price:.8f}",
                    f"{row.peg_deviation:.8f}",
                    f"{row.supply_stress_score:.8f}",
                    f"{row.score:.8f}",
                    row.status,
                    row.side,
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_peg_stress_md(
    rows: tuple[StablecoinPegStressRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Stablecoin Peg Stress\n\n")
        handle.write(
            "This screen looks for depeg/repeg candidates from DeFiLlama stablecoin prices and supply changes. "
            "It is a stress screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| symbol | name | status | price | peg deviation | week change USD | month change USD | score | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.name} | {row.status} | {row.price:.6f} | "
                f"{row.peg_deviation:.6f} | {row.week_change_usd:.0f} | "
                f"{row.month_change_usd:.0f} | {row.score:.4f} | {row.reason} |\n"
            )
    return output_path


def _build_row(row: StablecoinSupplyRow) -> StablecoinPegStressRow:
    peg_deviation = row.price - 1.0
    weekly_supply_change_pct = row.week_change_usd / row.current_supply_usd
    monthly_supply_change_pct = row.month_change_usd / row.current_supply_usd
    supply_stress_score = abs(weekly_supply_change_pct) * 10.0 + abs(monthly_supply_change_pct) * 3.0
    score = abs(peg_deviation) * 10_000.0 + min(supply_stress_score * 100.0, 25.0)
    status, side, reason = _status_side_reason(
        peg_deviation=peg_deviation,
        weekly_supply_change_pct=weekly_supply_change_pct,
        monthly_supply_change_pct=monthly_supply_change_pct,
    )
    return StablecoinPegStressRow(
        symbol=row.symbol,
        name=row.name,
        peg_type=row.peg_type,
        peg_mechanism=row.peg_mechanism,
        current_supply_usd=row.current_supply_usd,
        week_change_usd=row.week_change_usd,
        month_change_usd=row.month_change_usd,
        price=row.price,
        peg_deviation=peg_deviation,
        supply_stress_score=supply_stress_score,
        score=score,
        status=status,
        side=side,
        reason=reason,
        next_step=f"check {row.symbol} redemption route, liquidity, exchange depth, and repeated peg snapshots",
    )


def _status_side_reason(
    *,
    peg_deviation: float,
    weekly_supply_change_pct: float,
    monthly_supply_change_pct: float,
) -> tuple[str, str, str]:
    if peg_deviation <= -0.005:
        return "paper_depeg_repeg_watch", "watch_repeg_or_short_risk", "material below-peg deviation"
    if peg_deviation >= 0.005:
        return "paper_premium_mean_reversion_watch", "watch_premium_reversion", "material above-peg deviation"
    if abs(weekly_supply_change_pct) >= 0.05 or abs(monthly_supply_change_pct) >= 0.10:
        return "peg_supply_stress_watch", "none", "large supply change with near-peg price"
    return "watch", "none", "near peg without major supply stress"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=ROOT / "current_peg_stress_screen.csv")
    parser.add_argument("--markdown-output-path", type=Path, default=ROOT / "current_peg_stress_screen.md")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = build_peg_stress_rows(build_stablecoin_supply_rows(fetch_stablecoins()))
    write_peg_stress_csv(rows, output_path=args.output_path)
    write_peg_stress_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.symbol, f"price={row.price:.6f}", f"score={row.score:.4f}", row.reason)


if __name__ == "__main__":
    main()
