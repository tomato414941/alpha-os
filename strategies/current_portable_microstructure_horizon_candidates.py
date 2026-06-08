from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PortableMicrostructureHorizonCandidate:
    candidate_id: str
    asset: str
    status: str
    candidate_horizon: str
    rejected_horizon: str
    candidate_directional_return: str
    rejected_directional_return: str
    priority: float
    feature_state: str
    required_record: str
    next_step: str


def build_portable_microstructure_horizon_candidates(
    *,
    frontier_path: Path = ROOT / "current_portable_microstructure_feature_frontier.csv",
) -> tuple[PortableMicrostructureHorizonCandidate, ...]:
    rows = tuple(
        _candidate_row(row)
        for row in _read_rows(frontier_path)
        if row.get("status") in {"short_horizon_only", "delayed_or_reversal_support"}
    )
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_portable_microstructure_horizon_candidates_csv(
    rows: tuple[PortableMicrostructureHorizonCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "candidate_id",
                "asset",
                "status",
                "candidate_horizon",
                "rejected_horizon",
                "candidate_directional_return",
                "rejected_directional_return",
                "priority",
                "feature_state",
                "required_record",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.candidate_id,
                    row.asset,
                    row.status,
                    row.candidate_horizon,
                    row.rejected_horizon,
                    row.candidate_directional_return,
                    row.rejected_directional_return,
                    f"{row.priority:.8f}",
                    row.feature_state,
                    row.required_record,
                    row.next_step,
                )
            )
    return output_path


def write_portable_microstructure_horizon_candidates_md(
    rows: tuple[PortableMicrostructureHorizonCandidate, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Portable Microstructure Horizon Candidates\n\n")
        handle.write(
            "This splits portable microstructure rows by horizon. "
            "A row can be useful as a 15m feature, a delayed 1h feature, or a negative control; "
            "those cases must not be collapsed into one strategy claim.\n\n"
        )
        handle.write(
            "| candidate | asset | status | candidate horizon | rejected horizon | candidate return | rejected return | priority | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            handle.write(
                "| "
                f"{row.candidate_id} | "
                f"{row.asset} | "
                f"{row.status} | "
                f"{row.candidate_horizon} | "
                f"{row.rejected_horizon} | "
                f"{row.candidate_directional_return} | "
                f"{row.rejected_directional_return} | "
                f"{row.priority:.4f} | "
                f"{_escape(row.next_step)} |\n"
            )
    return output_path


def _candidate_row(row: dict[str, str]) -> PortableMicrostructureHorizonCandidate:
    status = row.get("status", "")
    asset = row.get("asset", "")
    if status == "short_horizon_only":
        candidate_horizon = "15m"
        rejected_horizon = "1h"
        candidate_return = row.get("directional_return_15m", "")
        rejected_return = row.get("directional_return_1h", "")
    else:
        candidate_horizon = "1h"
        rejected_horizon = "15m"
        candidate_return = row.get("directional_return_1h", "")
        rejected_return = row.get("directional_return_15m", "")
    return PortableMicrostructureHorizonCandidate(
        candidate_id=f"portable-micro-{asset.lower()}-{candidate_horizon}",
        asset=asset,
        status=status,
        candidate_horizon=candidate_horizon,
        rejected_horizon=rejected_horizon,
        candidate_directional_return=candidate_return,
        rejected_directional_return=rejected_return,
        priority=_priority(row=row, candidate_return=candidate_return, rejected_return=rejected_return),
        feature_state=row.get("feature_state", ""),
        required_record=(
            "repeat same feature on a fresh timestamp, execution-cost split, maker/taker path, "
            "queue/adverse-selection note, and negative-control horizon"
        ),
        next_step=_next_step(asset=asset, candidate_horizon=candidate_horizon, rejected_horizon=rejected_horizon),
    )


def _priority(*, row: dict[str, str], candidate_return: str, rejected_return: str) -> float:
    return (
        _float(row.get("priority"))
        + max(_float(candidate_return), 0.0) * 1_000.0
        + abs(min(_float(rejected_return), 0.0)) * 250.0
    )


def _next_step(*, asset: str, candidate_horizon: str, rejected_horizon: str) -> str:
    return (
        f"repeat {asset} portable microstructure feature at {candidate_horizon}; "
        f"treat {rejected_horizon} as a negative-control horizon until it repeats"
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier-path",
        type=Path,
        default=ROOT / "current_portable_microstructure_feature_frontier.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_portable_microstructure_horizon_candidates.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_portable_microstructure_horizon_candidates.md",
    )
    args = parser.parse_args()

    rows = build_portable_microstructure_horizon_candidates(frontier_path=args.frontier_path)
    write_portable_microstructure_horizon_candidates_csv(rows, output_path=args.output_path)
    write_portable_microstructure_horizon_candidates_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(row.candidate_id, row.status, f"{row.priority:.4f}", row.next_step)


if __name__ == "__main__":
    main()
