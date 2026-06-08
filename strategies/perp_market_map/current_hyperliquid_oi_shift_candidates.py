from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OpenInterestShiftCandidate:
    asset: str
    first_seen_at: str
    last_seen_at: str
    observations: int
    status: str
    side: str
    score: float
    open_interest_notional_first: float
    open_interest_notional_last: float
    open_interest_notional_change: float
    open_interest_notional_change_pct: float
    return_24h: float
    annualized_funding: float
    oi_volume_ratio: float
    impact_spread: float
    reason: str
    next_step: str


def build_open_interest_shift_candidates(
    *,
    input_path: Path = ROOT / "current_hyperliquid_dislocation_monitor_samples.csv",
) -> tuple[OpenInterestShiftCandidate, ...]:
    samples = _deduped_asset_samples(_read_rows(input_path))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in samples:
        grouped.setdefault(row["asset"], []).append(row)
    candidates = tuple(
        candidate
        for asset_rows in grouped.values()
        if (candidate := _candidate_from_asset_rows(tuple(asset_rows))) is not None
    )
    return tuple(sorted(candidates, key=lambda row: row.score, reverse=True))


def write_open_interest_shift_candidates_csv(
    rows: tuple[OpenInterestShiftCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "first_seen_at",
                "last_seen_at",
                "observations",
                "status",
                "side",
                "score",
                "open_interest_notional_first",
                "open_interest_notional_last",
                "open_interest_notional_change",
                "open_interest_notional_change_pct",
                "return_24h",
                "annualized_funding",
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
                    row.first_seen_at,
                    row.last_seen_at,
                    row.observations,
                    row.status,
                    row.side,
                    f"{row.score:.8f}",
                    f"{row.open_interest_notional_first:.8f}",
                    f"{row.open_interest_notional_last:.8f}",
                    f"{row.open_interest_notional_change:.8f}",
                    f"{row.open_interest_notional_change_pct:.8f}",
                    f"{row.return_24h:.8f}",
                    f"{row.annualized_funding:.8f}",
                    f"{row.oi_volume_ratio:.8f}",
                    f"{row.impact_spread:.12f}",
                    row.reason,
                    row.next_step,
                )
            )
    return output_path


def write_open_interest_shift_candidates_md(
    rows: tuple[OpenInterestShiftCandidate, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid OI Shift Candidates\n\n")
        handle.write(
            "This reads accumulated Hyperliquid dislocation monitor samples and "
            "looks for short-window open-interest notional shifts. It is a "
            "crowding/unwind candidate screen, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | status | side | score | obs | OI change | ret24 | funding ann | OI/vol | impact | reason |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.asset} | "
                f"{row.status} | "
                f"{row.side} | "
                f"{row.score:.4f} | "
                f"{row.observations} | "
                f"{row.open_interest_notional_change_pct:.4f} | "
                f"{row.return_24h:.4f} | "
                f"{row.annualized_funding:.4f} | "
                f"{row.oi_volume_ratio:.4f} | "
                f"{row.impact_spread:.6f} | "
                f"{row.reason} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "OI rising into a strong move can mean crowded continuation or a "
            "late squeeze setup. OI falling into a strong move can mean short "
            "covering, long liquidation, or crowded-risk decay. Both need forward "
            "labels, depth, and funding/fee costs before paper action.\n"
        )
    return output_path


def _deduped_asset_samples(rows: tuple[dict[str, str], ...]) -> tuple[dict[str, str], ...]:
    best_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("monitor_timestamp", ""), row.get("asset", ""))
        if not key[0] or not key[1]:
            continue
        existing = best_by_key.get(key)
        if existing is None or _float(row.get("score")) > _float(existing.get("score")):
            best_by_key[key] = row
    return tuple(sorted(best_by_key.values(), key=lambda row: (row["asset"], row["monitor_timestamp"])))


def _candidate_from_asset_rows(rows: tuple[dict[str, str], ...]) -> OpenInterestShiftCandidate | None:
    sorted_rows = tuple(sorted(rows, key=lambda row: row["monitor_timestamp"]))
    if len(sorted_rows) < 3:
        return None
    first = sorted_rows[0]
    last = sorted_rows[-1]
    first_oi = _float(first.get("open_interest_notional"))
    last_oi = _float(last.get("open_interest_notional"))
    if first_oi <= 0.0:
        return None
    oi_change = last_oi - first_oi
    oi_change_pct = (last_oi / first_oi) - 1.0
    return_24h = _float(last.get("return_24h"))
    annualized_funding = _float(last.get("annualized_funding"))
    oi_volume_ratio = _float(last.get("oi_volume_ratio"))
    impact_spread = _float(last.get("impact_spread"))
    if abs(oi_change_pct) < 0.01 or abs(return_24h) < 0.03:
        return None
    status, side, reason = _classify_candidate(
        oi_change_pct=oi_change_pct,
        return_24h=return_24h,
    )
    friction_penalty = impact_spread * 300.0
    score = (
        abs(oi_change_pct) * 900.0
        + abs(return_24h) * 80.0
        + min(oi_volume_ratio, 8.0)
        + abs(annualized_funding)
        - friction_penalty
    )
    if score <= 0.0:
        return None
    asset = last["asset"]
    return OpenInterestShiftCandidate(
        asset=asset,
        first_seen_at=first["monitor_timestamp"],
        last_seen_at=last["monitor_timestamp"],
        observations=len(sorted_rows),
        status=status,
        side=side,
        score=score,
        open_interest_notional_first=first_oi,
        open_interest_notional_last=last_oi,
        open_interest_notional_change=oi_change,
        open_interest_notional_change_pct=oi_change_pct,
        return_24h=return_24h,
        annualized_funding=annualized_funding,
        oi_volume_ratio=oi_volume_ratio,
        impact_spread=impact_spread,
        reason=reason,
        next_step=(
            f"label {asset} OI-shift candidate over 15m/1h/4h with funding, "
            "spread, depth, and failure-regime separation"
        ),
    )


def _classify_candidate(*, oi_change_pct: float, return_24h: float) -> tuple[str, str, str]:
    price_direction = 1 if return_24h > 0.0 else -1
    oi_direction = 1 if oi_change_pct > 0.0 else -1
    if price_direction > 0 and oi_direction > 0:
        return (
            "paper_oi_funding_crowding_watch",
            "long_perp",
            "OI notional is rising into an up move; test crowded continuation versus late-long squeeze risk",
        )
    if price_direction > 0 and oi_direction < 0:
        return (
            "paper_oi_unwind_watch",
            "context_only",
            "OI notional is falling into an up move; test short-cover exhaustion versus cleaner continuation",
        )
    if price_direction < 0 and oi_direction > 0:
        return (
            "paper_oi_funding_crowding_watch",
            "short_perp",
            "OI notional is rising into a down move; test crowded continuation versus late-short squeeze risk",
        )
    return (
        "paper_oi_unwind_watch",
        "context_only",
        "OI notional is falling into a down move; test liquidation exhaustion versus cleaner continuation",
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
        "--input-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_monitor_samples.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_oi_shift_candidates.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_oi_shift_candidates.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_open_interest_shift_candidates(input_path=args.input_path)
    write_open_interest_shift_candidates_csv(rows, output_path=args.output_path)
    write_open_interest_shift_candidates_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.status,
            row.side,
            f"score={row.score:.4f}",
            f"oi_change={row.open_interest_notional_change_pct:.4f}",
        )


if __name__ == "__main__":
    main()
