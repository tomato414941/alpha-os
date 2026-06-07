from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import log10
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HyperliquidDislocationCandidate:
    timestamp: str
    asset: str
    status: str
    side: str
    score: float
    return_24h: float
    annualized_funding: float
    mark_oracle_diff: float
    premium: float
    open_interest_notional: float
    day_notional_volume: float
    oi_volume_ratio: float
    impact_spread: float
    reason: str
    next_step: str


def build_hyperliquid_dislocation_candidates(
    *,
    snapshot_path: Path = ROOT / "current_hyperliquid_snapshot.csv",
) -> tuple[HyperliquidDislocationCandidate, ...]:
    candidates: list[HyperliquidDislocationCandidate] = []
    for row in _read_rows(snapshot_path):
        candidates.extend(_candidates_for_row(row))
    return tuple(sorted(candidates, key=lambda row: row.score, reverse=True))


def write_hyperliquid_dislocation_candidates_csv(
    rows: tuple[HyperliquidDislocationCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "timestamp",
                "status",
                "side",
                "score",
                "return_24h",
                "annualized_funding",
                "mark_oracle_diff",
                "premium",
                "open_interest_notional",
                "day_notional_volume",
                "oi_volume_ratio",
                "impact_spread",
                "reason",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.timestamp,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    f"{row.return_24h:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.mark_oracle_diff:.12f}",
                    f"{row.premium:.12f}",
                    f"{row.open_interest_notional:.8f}",
                    f"{row.day_notional_volume:.8f}",
                    f"{row.oi_volume_ratio:.8f}",
                    f"{row.impact_spread:.12f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_hyperliquid_dislocation_candidates_md(
    rows: tuple[HyperliquidDislocationCandidate, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Dislocation Candidates\n\n")
        handle.write(
            "This broadens the perp market map beyond carry reversion. It combines "
            "24h return, funding, mark/oracle dislocation, premium, OI/volume, and "
            "impact spread into current paper hypotheses. It is not a strategy or "
            "trade instruction.\n\n"
        )
        handle.write(
            "| asset | status | side | score | ret24 | funding ann | mark/oracle | premium | OI/vol | impact | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.asset} | "
                f"{row.status} | "
                f"{row.side} | "
                f"{row.score:.4f} | "
                f"{row.return_24h:.4f} | "
                f"{row.annualized_funding:.4f} | "
                f"{row.mark_oracle_diff:.6f} | "
                f"{row.premium:.6f} | "
                f"{row.oi_volume_ratio:.4f} | "
                f"{row.impact_spread:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The rows are hypotheses to label, not deployable trades. "
            "Continuation candidates ask whether a strong 24h move with crowding "
            "keeps moving. Reversal candidates ask whether the same crowding snaps "
            "back. Carry/premium candidates ask whether funding and mark/oracle "
            "dislocations decay after costs.\n"
        )
    return output_path


def _candidates_for_row(row: dict[str, str]) -> tuple[HyperliquidDislocationCandidate, ...]:
    asset = row["asset"]
    return_24h = _float(row.get("return_24h"))
    annualized_funding = _float(row["annualized_funding"])
    mark_oracle_diff = _float(row["mark_oracle_diff"])
    premium = _float(row["premium"])
    open_interest_notional = _float(row["open_interest_notional"])
    day_notional_volume = _float(row["day_notional_volume"])
    oi_volume_ratio = open_interest_notional / day_notional_volume if day_notional_volume > 0.0 else 0.0
    impact_spread = _float(row["impact_spread"])
    liquidity = log10(max(day_notional_volume, 1.0))
    friction_penalty = impact_spread * 250.0
    output: list[HyperliquidDislocationCandidate] = []

    if abs(return_24h) >= 0.05 and oi_volume_ratio >= 0.25 and day_notional_volume >= 250_000.0:
        direction = "long" if return_24h > 0.0 else "short"
        score = (
            abs(return_24h) * 120.0
            + min(oi_volume_ratio, 8.0)
            + liquidity / 2.0
            + abs(annualized_funding)
            - friction_penalty
        )
        output.append(
            _candidate(
                asset=asset,
                status="paper_crowded_momentum_continuation_candidate",
                side=f"{direction}_perp",
                score=score,
                row=row,
                reason="large 24h move with meaningful OI/volume; test continuation",
            )
        )
        output.append(
            _candidate(
                asset=asset,
                status="paper_crowded_momentum_reversal_candidate",
                side=f"{'short' if direction == 'long' else 'long'}_perp",
                score=score * 0.85,
                row=row,
                reason="large 24h move with meaningful OI/volume; test snapback",
            )
        )

    if abs(annualized_funding) >= 0.35 and day_notional_volume >= 100_000.0:
        side = "short_perp" if annualized_funding > 0.0 else "long_perp"
        score = (
            abs(annualized_funding) * liquidity
            + abs(mark_oracle_diff) * 100.0
            + min(oi_volume_ratio, 8.0)
            - friction_penalty
        )
        output.append(
            _candidate(
                asset=asset,
                status="paper_extreme_funding_carry_candidate",
                side=side,
                score=score,
                row=row,
                reason="extreme funding may pay carry or precede crowded unwind",
            )
        )

    if abs(mark_oracle_diff) >= 0.0015 and day_notional_volume >= 100_000.0:
        side = "short_perp" if mark_oracle_diff > 0.0 else "long_perp"
        score = (
            abs(mark_oracle_diff) * 1_000.0
            + abs(premium) * 250.0
            + liquidity / 2.0
            + min(oi_volume_ratio, 8.0)
            - friction_penalty
        )
        output.append(
            _candidate(
                asset=asset,
                status="paper_mark_oracle_reversion_candidate",
                side=side,
                score=score,
                row=row,
                reason="mark/oracle dislocation is large enough to label reversion",
            )
        )

    return tuple(candidate for candidate in output if candidate.score > 0.0)


def _candidate(
    *,
    asset: str,
    status: str,
    side: str,
    score: float,
    row: dict[str, str],
    reason: str,
) -> HyperliquidDislocationCandidate:
    open_interest_notional = _float(row["open_interest_notional"])
    day_notional_volume = _float(row["day_notional_volume"])
    return HyperliquidDislocationCandidate(
        timestamp=row.get("timestamp", ""),
        asset=asset,
        status=status,
        side=side,
        score=score,
        return_24h=_float(row.get("return_24h")),
        annualized_funding=_float(row["annualized_funding"]),
        mark_oracle_diff=_float(row["mark_oracle_diff"]),
        premium=_float(row["premium"]),
        open_interest_notional=open_interest_notional,
        day_notional_volume=day_notional_volume,
        oi_volume_ratio=open_interest_notional / day_notional_volume if day_notional_volume > 0.0 else 0.0,
        impact_spread=_float(row["impact_spread"]),
        reason=reason,
        next_step=f"label {asset} {status} over 15m/1h/4h with funding, spread, and depth costs",
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=ROOT / "current_hyperliquid_snapshot.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_candidates.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_candidates.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_hyperliquid_dislocation_candidates(snapshot_path=args.snapshot_path)
    write_hyperliquid_dislocation_candidates_csv(rows, output_path=args.output_path)
    write_hyperliquid_dislocation_candidates_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.status, row.asset, row.side, f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
