from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FollowupQueueRow:
    priority: float
    asset: str
    source: str
    followup_type: str
    action: str
    evidence: str
    next_test: str


def build_followup_queue_rows() -> tuple[FollowupQueueRow, ...]:
    rows: list[FollowupQueueRow] = []
    rows.extend(_clean_candidate_rows())
    rows.extend(_source_conflict_rows())
    rows.extend(_family_repeat_rows())
    return tuple(sorted(rows, key=lambda row: row.priority, reverse=True))


def write_followup_queue_csv(
    rows: tuple[FollowupQueueRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "priority",
                "asset",
                "source",
                "followup_type",
                "action",
                "evidence",
                "next_test",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    f"{row.priority:.4f}",
                    row.asset,
                    row.source,
                    row.followup_type,
                    row.action,
                    row.evidence,
                    row.next_test,
                )
            )
    return output_path


def write_followup_queue_md(
    rows: tuple[FollowupQueueRow, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up Queue\n\n")
        handle.write(
            "This queue turns current labels into repeatable next observations. "
            "It is not a trading instruction; it is a work queue for finding "
            "which alpha source is real.\n\n"
        )
        handle.write(
            "| priority | asset | source | type | action | evidence | next test |\n"
        )
        handle.write("| ---: | --- | --- | --- | --- | --- | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.priority:.4f} | "
                f"{row.asset} | "
                f"{row.source} | "
                f"{row.followup_type} | "
                f"{row.action} | "
                f"{row.evidence} | "
                f"{row.next_test} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The queue deliberately separates source-specific repeats from "
            "cross-lane aggregation. A candidate should graduate only after the "
            "same source survives repeated labels and rough cost checks.\n"
        )
    return output_path


def _clean_candidate_rows() -> tuple[FollowupQueueRow, ...]:
    rows: list[FollowupQueueRow] = []
    for row in _read_rows(ROOT / "current_cross_lane_candidate_review.csv"):
        positives = _split(row.get("positive_labels", ""))
        negatives = _split(row.get("negative_labels", ""))
        if not positives or negatives:
            continue
        rows.append(
            FollowupQueueRow(
                priority=float(row.get("lead_score") or "0") + len(positives),
                asset=row["asset"],
                source=_sources_from_labels(positives),
                followup_type="clean_candidate_repeat",
                action="repeat_supported_candidate",
                evidence=";".join(positives),
                next_test="repeat the same labels on fresh samples and add rough costs",
            )
        )
    return tuple(rows)


def _source_conflict_rows() -> tuple[FollowupQueueRow, ...]:
    rows: list[FollowupQueueRow] = []
    for row in _read_rows(ROOT / "current_source_conflict_review.csv"):
        positives = _split(row.get("positive_sources", ""))
        rows.append(
            FollowupQueueRow(
                priority=float(row.get("score") or "0") + len(positives) * 0.5,
                asset=row["asset"],
                source=row.get("positive_sources", ""),
                followup_type="source_isolation",
                action=row.get("action", ""),
                evidence=f"positive={row.get('positive_sources', '')};negative={row.get('negative_sources', '')}",
                next_test=row.get("next_test", ""),
            )
        )
    return tuple(rows)


def _family_repeat_rows() -> tuple[FollowupQueueRow, ...]:
    rows: list[FollowupQueueRow] = []
    for row in _read_rows(ROOT / "current_signal_family_review.csv"):
        if row.get("note") != "supported by first labels":
            continue
        rows.append(
            FollowupQueueRow(
                priority=float(row.get("support_score") or "0") + float(row.get("hit_rate_15m") or "0"),
                asset="*",
                source=row["family"],
                followup_type="family_repeat",
                action="repeat_supported_family",
                evidence=(
                    f"cov15={row.get('coverage_15m', '')};"
                    f"mean15={row.get('mean_label_15m', '')};"
                    f"hit15={row.get('hit_rate_15m', '')}"
                ),
                next_test="collect more labels from this family and compare against neutral/cost baselines",
            )
        )
    return tuple(rows)


def _sources_from_labels(labels: tuple[str, ...]) -> str:
    sources: list[str] = []
    for label in labels:
        source = _source_for_label(label)
        if source not in sources:
            sources.append(source)
    return ";".join(sources)


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
        "--output-path",
        type=Path,
        default=ROOT / "current_followup_queue.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_followup_queue.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = build_followup_queue_rows()
    write_followup_queue_csv(rows, output_path=args.output_path)
    write_followup_queue_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.asset,
            row.source,
            row.followup_type,
            f"priority={row.priority:.4f}",
        )


if __name__ == "__main__":
    main()
