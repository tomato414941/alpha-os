from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class ValidatedCrowdingReversionRow:
    asset: str
    action: str
    status: str
    validation_score: float
    monitor_observations: int
    mean_crowding_score: float
    min_crowding_score: float
    mean_annualized_funding: float
    mean_mark_oracle_diff: float
    mean_oi_volume_ratio: float
    mean_impact_spread: float
    label_observations: int
    coverage_15m: int
    coverage_1h: int
    mean_directional_return_15m: float
    mean_directional_return_1h: float
    positive_directional_15m_rate: float
    positive_directional_1h_rate: float
    next_step: str


def build_validated_crowding_reversion_rows(
    *,
    monitor_path: Path = ROOT / "current_crowding_reversion_monitor_summary.csv",
    label_path: Path = STRATEGIES_ROOT / "candidate_validation" / "current_hl_signal_forward_label_summary.csv",
) -> tuple[ValidatedCrowdingReversionRow, ...]:
    monitor_rows = {
        (row.get("asset", ""), row.get("action", "")): row
        for row in _read_rows(monitor_path)
    }
    labels = tuple(
        row
        for row in _read_rows(label_path)
        if row.get("source") == "perp_carry_reversion"
    )
    rows = tuple(
        _build_row(label=row, monitor=monitor_rows.get((row.get("asset", ""), row.get("action", "")), {}))
        for row in labels
    )
    return tuple(sorted(rows, key=lambda row: row.validation_score, reverse=True))


def write_validated_crowding_reversion_csv(
    rows: tuple[ValidatedCrowdingReversionRow, ...],
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
                "status",
                "validation_score",
                "monitor_observations",
                "mean_crowding_score",
                "min_crowding_score",
                "mean_annualized_funding",
                "mean_mark_oracle_diff",
                "mean_oi_volume_ratio",
                "mean_impact_spread",
                "label_observations",
                "coverage_15m",
                "coverage_1h",
                "mean_directional_return_15m",
                "mean_directional_return_1h",
                "positive_directional_15m_rate",
                "positive_directional_1h_rate",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.status,
                    f"{row.validation_score:.8f}",
                    row.monitor_observations,
                    f"{row.mean_crowding_score:.8f}",
                    f"{row.min_crowding_score:.8f}",
                    f"{row.mean_annualized_funding:.8f}",
                    f"{row.mean_mark_oracle_diff:.12f}",
                    f"{row.mean_oi_volume_ratio:.8f}",
                    f"{row.mean_impact_spread:.12f}",
                    row.label_observations,
                    row.coverage_15m,
                    row.coverage_1h,
                    f"{row.mean_directional_return_15m:.8f}",
                    f"{row.mean_directional_return_1h:.8f}",
                    f"{row.positive_directional_15m_rate:.8f}",
                    f"{row.positive_directional_1h_rate:.8f}",
                    row.next_step,
                )
            )
    return output_path


def write_validated_crowding_reversion_md(
    rows: tuple[ValidatedCrowdingReversionRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Crowding Reversion Validated Candidates\n\n")
        handle.write(
            "This joins short-window persistence with directional forward labels. "
            "It is a candidate-ranking view, not a trade instruction.\n\n"
        )
        handle.write(
            "| asset | action | status | score | monitor obs | label obs | mean dir 15m | mean dir 1h | hit15 | hit1h | funding | impact | next step |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.asset} | {row.action} | {row.status} | {row.validation_score:.4f} | "
                f"{row.monitor_observations} | {row.label_observations} | "
                f"{row.mean_directional_return_15m:.6f} | {row.mean_directional_return_1h:.6f} | "
                f"{row.positive_directional_15m_rate:.4f} | {row.positive_directional_1h_rate:.4f} | "
                f"{row.mean_annualized_funding:.4f} | {row.mean_impact_spread:.6f} | {_escape(row.next_step)} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_validated_carry_reversion_candidate` means the signal persisted and "
            "the directional label was positive over both 15m and 1h in the current small sample. "
            "It still needs a larger repeat sample, funding PnL, fees, spread, and stop behavior.\n"
        )
    return output_path


def _build_row(
    *,
    label: dict[str, str],
    monitor: dict[str, str],
) -> ValidatedCrowdingReversionRow:
    action = label.get("action", "")
    mean_15m = _float(label.get("mean_directional_return_15m"))
    mean_1h = _float(label.get("mean_directional_return_1h"))
    hit_15m = _float(label.get("positive_directional_15m_rate"))
    hit_1h = _float(label.get("positive_directional_1h_rate"))
    coverage_15m = _int(label.get("coverage_15m"))
    coverage_1h = _int(label.get("coverage_1h"))
    monitor_observations = _int(monitor.get("observations"))
    status = _status(
        mean_15m=mean_15m,
        mean_1h=mean_1h,
        hit_15m=hit_15m,
        hit_1h=hit_1h,
        coverage_15m=coverage_15m,
        coverage_1h=coverage_1h,
        monitor_observations=monitor_observations,
    )
    asset = label.get("asset", "")
    return ValidatedCrowdingReversionRow(
        asset=asset,
        action=action,
        status=status,
        validation_score=_validation_score(
            status=status,
            mean_15m=mean_15m,
            mean_1h=mean_1h,
            hit_15m=hit_15m,
            hit_1h=hit_1h,
            monitor_score=_float(monitor.get("mean_score")),
            monitor_observations=monitor_observations,
            coverage_1h=coverage_1h,
            impact_spread=_float(monitor.get("mean_impact_spread")),
        ),
        monitor_observations=monitor_observations,
        mean_crowding_score=_float(monitor.get("mean_score")),
        min_crowding_score=_float(monitor.get("min_score")),
        mean_annualized_funding=_float(monitor.get("mean_annualized_funding")),
        mean_mark_oracle_diff=_float(monitor.get("mean_mark_oracle_diff")),
        mean_oi_volume_ratio=_float(monitor.get("mean_oi_volume_ratio")),
        mean_impact_spread=_float(monitor.get("mean_impact_spread")),
        label_observations=_int(label.get("observations")),
        coverage_15m=coverage_15m,
        coverage_1h=coverage_1h,
        mean_directional_return_15m=mean_15m,
        mean_directional_return_1h=mean_1h,
        positive_directional_15m_rate=hit_15m,
        positive_directional_1h_rate=hit_1h,
        next_step=_next_step(asset=asset, action=action, status=status),
    )


def _status(
    *,
    mean_15m: float,
    mean_1h: float,
    hit_15m: float,
    hit_1h: float,
    coverage_15m: int,
    coverage_1h: int,
    monitor_observations: int,
) -> str:
    if coverage_15m >= 4 and coverage_1h >= 4 and monitor_observations >= 4:
        if mean_15m > 0.0 and mean_1h > 0.0 and hit_15m >= 0.75 and hit_1h >= 0.75:
            return "paper_validated_carry_reversion_candidate"
        if mean_15m <= 0.0 and mean_1h > 0.0 and hit_1h >= 0.75:
            return "paper_delayed_carry_reversion_watch"
        if mean_15m < 0.0 and mean_1h < 0.0:
            return "paper_carry_reversion_reject"
    return "paper_carry_reversion_needs_more_labels"


def _validation_score(
    *,
    status: str,
    mean_15m: float,
    mean_1h: float,
    hit_15m: float,
    hit_1h: float,
    monitor_score: float,
    monitor_observations: int,
    coverage_1h: int,
    impact_spread: float,
) -> float:
    status_bonus = {
        "paper_validated_carry_reversion_candidate": 100.0,
        "paper_delayed_carry_reversion_watch": 80.0,
        "paper_carry_reversion_needs_more_labels": 55.0,
        "paper_carry_reversion_reject": 20.0,
    }.get(status, 40.0)
    return (
        status_bonus
        + min(monitor_score, 15.0)
        + min(monitor_observations, 6) * 1.5
        + min(coverage_1h, 6) * 1.5
        + (mean_15m * 1000.0)
        + (mean_1h * 1000.0)
        + (hit_15m + hit_1h) * 5.0
        - impact_spread * 100.0
    )


def _next_step(*, asset: str, action: str, status: str) -> str:
    if status == "paper_validated_carry_reversion_candidate":
        return f"repeat {asset} {action} over fresh windows, then add funding PnL, fees, spread, and stop behavior"
    if status == "paper_delayed_carry_reversion_watch":
        return f"check whether {asset} {action} needs delayed entry rather than immediate entry"
    if status == "paper_carry_reversion_reject":
        return f"do not promote {asset} {action}; investigate whether the opposite side or no-trade rule is better"
    return f"collect more {asset} {action} labels before treating it as an alpha candidate"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _int(value: str | None) -> int:
    return int(value) if value else 0


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monitor-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_monitor_summary.csv",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=STRATEGIES_ROOT / "candidate_validation" / "current_hl_signal_forward_label_summary.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_validated_candidates.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_crowding_reversion_validated_candidates.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_validated_crowding_reversion_rows(
        monitor_path=args.monitor_path,
        label_path=args.label_path,
    )
    write_validated_crowding_reversion_csv(rows, output_path=args.output_path)
    write_validated_crowding_reversion_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            row.status,
            f"score={row.validation_score:.4f}",
            f"dir1h={row.mean_directional_return_1h:.4f}",
        )


if __name__ == "__main__":
    main()
