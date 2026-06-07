from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FollowupRepeatSummaryRow:
    group_type: str
    group_key: str
    labeled_rows: int
    pending_rows: int
    hit_rate_15m: float | None
    mean_dir15: float | None
    min_dir15: float | None
    max_dir15: float | None
    action: str
    evidence: str


def build_followup_repeat_history_summary_rows(
    *,
    label_path: Path = ROOT / "current_followup_repeat_history_labels.csv",
) -> tuple[FollowupRepeatSummaryRow, ...]:
    rows = _read_rows(label_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        for group_type, group_key in _group_keys(row):
            grouped.setdefault((group_type, group_key), []).append(row)
    summary_rows = tuple(
        _summary_row(group_type=group_type, group_key=group_key, rows=group_rows)
        for (group_type, group_key), group_rows in grouped.items()
    )
    return tuple(sorted(summary_rows, key=_sort_key, reverse=True))


def write_followup_repeat_history_summary_csv(
    rows: tuple[FollowupRepeatSummaryRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "group_type",
                "group_key",
                "labeled_rows",
                "pending_rows",
                "hit_rate_15m",
                "mean_dir15",
                "min_dir15",
                "max_dir15",
                "action",
                "evidence",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.group_type,
                    row.group_key,
                    row.labeled_rows,
                    row.pending_rows,
                    "" if row.hit_rate_15m is None else f"{row.hit_rate_15m:.6f}",
                    "" if row.mean_dir15 is None else f"{row.mean_dir15:.8f}",
                    "" if row.min_dir15 is None else f"{row.min_dir15:.8f}",
                    "" if row.max_dir15 is None else f"{row.max_dir15:.8f}",
                    row.action,
                    row.evidence,
                )
            )
    return output_path


def write_followup_repeat_history_summary_md(
    rows: tuple[FollowupRepeatSummaryRow, ...],
    *,
    output_path: Path,
    top: int = 50,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repeat_priority = tuple(row for row in rows if row.action == "repeat_priority")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Follow-Up Repeat History Summary\n\n")
        handle.write(
            "This aggregates stored repeat labels without blending source meaning. "
            "Rows with pending observations should be rerun after the 15m horizon matures.\n\n"
        )
        handle.write(f"- total groups: `{len(rows)}`\n")
        handle.write(f"- repeat-priority groups: `{len(repeat_priority)}`\n\n")
        handle.write(
            "| group type | group | labeled | pending | hit 15m | mean dir15 | min dir15 | max dir15 | action | evidence |\n"
        )
        handle.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
        )
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.group_type} | "
                f"{row.group_key} | "
                f"{row.labeled_rows} | "
                f"{row.pending_rows} | "
                f"{'' if row.hit_rate_15m is None else f'{row.hit_rate_15m:.3f}'} | "
                f"{'' if row.mean_dir15 is None else f'{row.mean_dir15:.6f}'} | "
                f"{'' if row.min_dir15 is None else f'{row.min_dir15:.6f}'} | "
                f"{'' if row.max_dir15 is None else f'{row.max_dir15:.6f}'} | "
                f"{row.action} | "
                f"{row.evidence} |\n"
            )
    return output_path


def _summary_row(
    *,
    group_type: str,
    group_key: str,
    rows: list[dict[str, str]],
) -> FollowupRepeatSummaryRow:
    labeled_values = tuple(
        float(row["directional_return_15m"])
        for row in rows
        if row.get("directional_return_15m", "") != ""
    )
    pending_rows = sum(1 for row in rows if row.get("directional_return_15m", "") == "")
    hit_rate = (
        None
        if not labeled_values
        else sum(1 for value in labeled_values if value > 0.0) / len(labeled_values)
    )
    mean_dir15 = None if not labeled_values else sum(labeled_values) / len(labeled_values)
    min_dir15 = None if not labeled_values else min(labeled_values)
    max_dir15 = None if not labeled_values else max(labeled_values)
    action = _action(
        labeled_rows=len(labeled_values),
        pending_rows=pending_rows,
        hit_rate=hit_rate,
        mean_dir15=mean_dir15,
        min_dir15=min_dir15,
        max_dir15=max_dir15,
    )
    evidence = _evidence(rows=rows, labeled_values=labeled_values)
    return FollowupRepeatSummaryRow(
        group_type=group_type,
        group_key=group_key,
        labeled_rows=len(labeled_values),
        pending_rows=pending_rows,
        hit_rate_15m=hit_rate,
        mean_dir15=mean_dir15,
        min_dir15=min_dir15,
        max_dir15=max_dir15,
        action=action,
        evidence=evidence,
    )


def _action(
    *,
    labeled_rows: int,
    pending_rows: int,
    hit_rate: float | None,
    mean_dir15: float | None,
    min_dir15: float | None,
    max_dir15: float | None,
) -> str:
    if labeled_rows == 0:
        return "wait_for_label"
    if labeled_rows < 2 and pending_rows > 0:
        return "wait_for_second_label"
    if labeled_rows < 2:
        return "collect_repeat"
    if hit_rate is not None and mean_dir15 is not None:
        if hit_rate >= 0.75 and mean_dir15 >= 0.001:
            return "repeat_priority"
        if hit_rate <= 0.25 and mean_dir15 < 0.0:
            return "deprioritize"
        if min_dir15 is not None and max_dir15 is not None and min_dir15 < 0.0 < max_dir15:
            return "mixed_continue_sampling"
        if mean_dir15 > 0.0:
            return "keep_sampling"
    return "deprioritize"


def _evidence(*, rows: list[dict[str, str]], labeled_values: tuple[float, ...]) -> str:
    labels = tuple(
        f"{row.get('venue', '')}/{row.get('asset', '')}/{row.get('source', '')}"
        for row in rows
        if row.get("directional_return_15m", "") != ""
    )
    if not labeled_values:
        return "all rows pending"
    return (
        f"examples={';'.join(labels[:3])}; "
        f"pending={sum(1 for row in rows if row.get('directional_return_15m', '') == '')}"
    )


def _group_keys(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
    venue = row.get("venue", "")
    asset = row.get("asset", "")
    source = row.get("source", "")
    return (
        ("source", source),
        ("venue_source", f"{venue}/{source}"),
        ("asset_source", f"{asset}/{source}"),
        ("venue_asset_source", f"{venue}/{asset}/{source}"),
    )


def _sort_key(row: FollowupRepeatSummaryRow) -> tuple[int, float, int, float]:
    action_rank = {
        "repeat_priority": 5,
        "mixed_continue_sampling": 4,
        "keep_sampling": 3,
        "wait_for_second_label": 2,
        "collect_repeat": 1,
        "wait_for_label": 0,
        "deprioritize": -1,
    }.get(row.action, 0)
    return (
        action_rank,
        row.mean_dir15 or -1.0,
        row.labeled_rows,
        row.hit_rate_15m or 0.0,
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-path",
        type=Path,
        default=ROOT / "current_followup_repeat_history_labels.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_followup_repeat_history_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_followup_repeat_history_summary.md",
    )
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = build_followup_repeat_history_summary_rows(label_path=args.label_path)
    write_followup_repeat_history_summary_csv(rows, output_path=args.output_path)
    write_followup_repeat_history_summary_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.group_type,
            row.group_key,
            row.action,
            f"n={row.labeled_rows}",
            f"pending={row.pending_rows}",
            f"mean15={'' if row.mean_dir15 is None else f'{row.mean_dir15:.4f}'}",
        )


if __name__ == "__main__":
    main()
