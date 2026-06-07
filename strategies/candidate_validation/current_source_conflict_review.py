from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SourceConflictRow:
    asset: str
    score: float
    lanes: str
    positive_sources: str
    negative_sources: str
    positive_count: int
    negative_count: int
    action: str
    next_test: str


def build_source_conflict_rows(
    *,
    input_path: Path = ROOT / "current_cross_lane_candidate_review.csv",
) -> tuple[SourceConflictRow, ...]:
    rows = tuple(_build_row(row) for row in _read_rows(input_path))
    mixed_rows = tuple(row for row in rows if row.positive_count > 0 and row.negative_count > 0)
    return tuple(
        sorted(
            mixed_rows,
            key=lambda row: (row.score, row.positive_count, -row.negative_count),
            reverse=True,
        )
    )


def write_source_conflict_csv(
    rows: tuple[SourceConflictRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "score",
                "lanes",
                "positive_sources",
                "negative_sources",
                "positive_count",
                "negative_count",
                "action",
                "next_test",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    f"{row.score:.4f}",
                    row.lanes,
                    row.positive_sources,
                    row.negative_sources,
                    row.positive_count,
                    row.negative_count,
                    row.action,
                    row.next_test,
                )
            )
    return output_path


def write_source_conflict_md(
    rows: tuple[SourceConflictRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Source Conflict Review\n\n")
        handle.write(
            "This isolates mixed-evidence candidates by source. It asks which "
            "signal source should be repeated or separated next, not whether the "
            "asset is deployable.\n\n"
        )
        handle.write(
            "| asset | score | positives | negatives | action | next test |\n"
        )
        handle.write("| --- | ---: | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.score:.4f} | "
                f"{row.positive_sources} | "
                f"{row.negative_sources} | "
                f"{row.action} | "
                f"{row.next_test} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "A mixed candidate is not a failure. It means the project should stop "
            "averaging incompatible sources together and repeat the source that is "
            "actually carrying the positive label.\n"
        )
    return output_path


def _build_row(row: dict[str, str]) -> SourceConflictRow:
    positives = _split(row.get("positive_labels", ""))
    negatives = _split(row.get("negative_labels", ""))
    positive_sources = tuple(_source_for_label(label) for label in positives)
    negative_sources = tuple(_source_for_label(label) for label in negatives)
    return SourceConflictRow(
        asset=row.get("asset", ""),
        score=float(row.get("lead_score") or "0"),
        lanes=row.get("lanes", ""),
        positive_sources=";".join(positive_sources),
        negative_sources=";".join(negative_sources),
        positive_count=len(positives),
        negative_count=len(negatives),
        action=_action(positive_sources=positive_sources, negative_sources=negative_sources),
        next_test=_next_test(positive_sources=positive_sources, negative_sources=negative_sources),
    )


def _action(
    *,
    positive_sources: tuple[str, ...],
    negative_sources: tuple[str, ...],
) -> str:
    if "sector_rotation" in positive_sources and "l2_imbalance" in negative_sources:
        return "separate_sector_from_l2"
    if "liquidation" in positive_sources and "okx_pressure" in negative_sources:
        return "repeat_liquidation_not_pressure"
    if "l2_imbalance" in positive_sources and "okx_pressure" in negative_sources:
        return "repeat_l2_not_pressure"
    if "hl_candidate" in positive_sources and "sector_rotation" in negative_sources:
        return "separate_carry_from_sector"
    return "isolate_positive_source"


def _next_test(
    *,
    positive_sources: tuple[str, ...],
    negative_sources: tuple[str, ...],
) -> str:
    if "sector_rotation" in positive_sources:
        return "repeat sector labels with category membership and costs before mixing with other sources"
    if "liquidation" in positive_sources:
        return "repeat fresh liquidation labels and ignore conflicting carry-pressure rows for this test"
    if "l2_imbalance" in positive_sources:
        return "repeat L2 labels with fill/adverse-selection assumptions before using as directional alpha"
    if "hl_candidate" in positive_sources:
        return "repeat the original candidate family and keep unrelated negative sources out of the decision"
    return "repeat the positive label source separately"


def _source_for_label(label: str) -> str:
    if label.startswith("hl15="):
        return "hl_candidate"
    if label.startswith("okx_pressure15="):
        return "okx_pressure"
    if label.startswith("liq_cont15="):
        return "liquidation"
    if label.startswith("l2_imbalance15="):
        return "l2_imbalance"
    if label.startswith("sector15="):
        return "sector_rotation"
    return "unknown"


def _split(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(";") if part)


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_cross_lane_candidate_review.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_source_conflict_review.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_source_conflict_review.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_source_conflict_rows(input_path=args.input_path)
    write_source_conflict_csv(rows, output_path=args.output_path)
    write_source_conflict_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.action,
            f"positive={row.positive_sources}",
            f"negative={row.negative_sources}",
        )


if __name__ == "__main__":
    main()
