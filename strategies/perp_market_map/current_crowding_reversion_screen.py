from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import log10
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CrowdingReversionRow:
    asset: str
    action: str
    annualized_funding: float
    mark_oracle_diff: float
    premium: float
    open_interest: float
    open_interest_notional: float
    day_notional_volume: float
    oi_volume_ratio: float
    impact_spread: float
    carry_reversion_score: float
    reason: str


def build_crowding_reversion_rows(
    *,
    snapshot_path: Path = ROOT / "current_hyperliquid_snapshot.csv",
) -> tuple[CrowdingReversionRow, ...]:
    with snapshot_path.open(newline="", encoding="utf-8") as handle:
        source_rows = tuple(csv.DictReader(handle))
    rows = tuple(_build_row(row) for row in source_rows)
    candidates = tuple(row for row in rows if row.action != "ignore")
    return tuple(sorted(candidates, key=lambda row: row.carry_reversion_score, reverse=True))


def write_crowding_reversion_csv(
    rows: tuple[CrowdingReversionRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "action",
                "annualized_funding",
                "mark_oracle_diff",
                "premium",
                "open_interest",
                "open_interest_notional",
                "day_notional_volume",
                "oi_volume_ratio",
                "impact_spread",
                "carry_reversion_score",
                "reason",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    f"{row.annualized_funding:.8f}",
                    f"{row.mark_oracle_diff:.12f}",
                    f"{row.premium:.12f}",
                    f"{row.open_interest:.8f}",
                    f"{row.open_interest_notional:.8f}",
                    f"{row.day_notional_volume:.8f}",
                    f"{row.oi_volume_ratio:.8f}",
                    f"{row.impact_spread:.12f}",
                    f"{row.carry_reversion_score:.8f}",
                    row.reason,
                )
            )
    return output_path


def write_crowding_reversion_md(
    rows: tuple[CrowdingReversionRow, ...],
    *,
    output_path: Path,
    top: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowding Reversion Screen\n\n")
        handle.write(
            "This screen looks for perp states where funding carry and mark/oracle "
            "reversion point in the same direction. It is not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | annualized funding | mark/oracle | premium | OI/volume | impact spread | score | reason |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.annualized_funding:.6f} | "
                f"{row.mark_oracle_diff:.6f} | "
                f"{row.premium:.6f} | "
                f"{row.oi_volume_ratio:.6f} | "
                f"{row.impact_spread:.6f} | "
                f"{row.carry_reversion_score:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`long_carry_reversion_watch` means long perp receives funding while the "
            "mark is below oracle. `short_carry_reversion_watch` means short perp "
            "receives funding while the mark is above oracle. OI/volume is a crowding "
            "proxy, not proof of forced liquidations or future returns.\n"
        )
    return output_path


def _build_row(row: dict[str, str]) -> CrowdingReversionRow:
    annualized_funding = _float(row["annualized_funding"])
    mark_oracle_diff = _float(row["mark_oracle_diff"])
    premium = _float(row["premium"])
    open_interest = _float(row["open_interest"])
    open_interest_notional = _float(row["open_interest_notional"])
    day_notional_volume = _float(row["day_notional_volume"])
    impact_spread = _float(row["impact_spread"])
    oi_volume_ratio = (
        open_interest_notional / day_notional_volume if day_notional_volume > 0.0 else 0.0
    )
    action = _action(
        annualized_funding=annualized_funding,
        mark_oracle_diff=mark_oracle_diff,
    )
    return CrowdingReversionRow(
        asset=row["asset"],
        action=action,
        annualized_funding=annualized_funding,
        mark_oracle_diff=mark_oracle_diff,
        premium=premium,
        open_interest=open_interest,
        open_interest_notional=open_interest_notional,
        day_notional_volume=day_notional_volume,
        oi_volume_ratio=oi_volume_ratio,
        impact_spread=impact_spread,
        carry_reversion_score=_score(
            action=action,
            annualized_funding=annualized_funding,
            mark_oracle_diff=mark_oracle_diff,
            premium=premium,
            day_notional_volume=day_notional_volume,
            oi_volume_ratio=oi_volume_ratio,
            impact_spread=impact_spread,
        ),
        reason=_reason(action),
    )


def _action(*, annualized_funding: float, mark_oracle_diff: float) -> str:
    if annualized_funding < 0.0 and mark_oracle_diff < 0.0:
        return "long_carry_reversion_watch"
    if annualized_funding > 0.0 and mark_oracle_diff > 0.0:
        return "short_carry_reversion_watch"
    return "ignore"


def _score(
    *,
    action: str,
    annualized_funding: float,
    mark_oracle_diff: float,
    premium: float,
    day_notional_volume: float,
    oi_volume_ratio: float,
    impact_spread: float,
) -> float:
    if action == "ignore":
        return float("-inf")
    liquidity = log10(max(day_notional_volume, 1.0))
    carry = abs(annualized_funding)
    reversion = abs(mark_oracle_diff) * 100.0
    premium_alignment = abs(premium) * 25.0
    crowding = min(oi_volume_ratio, 10.0)
    friction_penalty = impact_spread * 100.0
    return (carry * liquidity) + reversion + premium_alignment + crowding - friction_penalty


def _reason(action: str) -> str:
    if action == "long_carry_reversion_watch":
        return "long perp receives funding and mark is below oracle"
    if action == "short_carry_reversion_watch":
        return "short perp receives funding and mark is above oracle"
    return "carry and mark/oracle reversion do not align"


def _float(value: str) -> float:
    return float(value or "0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=ROOT / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_screen.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_screen.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_crowding_reversion_rows(snapshot_path=args.snapshot_path)
    write_crowding_reversion_csv(rows, output_path=args.csv_output_path)
    write_crowding_reversion_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"funding={row.annualized_funding:.4f}",
            f"mark_oracle={row.mark_oracle_diff:.6f}",
            f"score={row.carry_reversion_score:.4f}",
        )


if __name__ == "__main__":
    main()
