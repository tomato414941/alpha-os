from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DislocationLabelHistorySummary:
    asset: str
    status: str
    side: str
    observations: int
    covered_15m: int
    covered_1h: int
    covered_4h: int
    win_15m: int
    win_1h: int
    win_4h: int
    mean_net_15m_bps: float
    mean_net_1h_bps: float
    mean_net_4h_bps: float
    best_net_15m_bps: float
    best_net_1h_bps: float
    best_net_4h_bps: float
    history_action: str


def merge_label_history(
    *,
    current_label_path: Path = ROOT / "current_hyperliquid_dislocation_forward_labels.csv",
    history_path: Path = ROOT / "current_hyperliquid_dislocation_label_history.csv",
) -> tuple[dict[str, str], ...]:
    existing = _read_rows(history_path)
    current = _read_rows(current_label_path)
    merged_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in (*existing, *current):
        key = (row.get("timestamp", ""), row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        if not all(key):
            continue
        previous = merged_by_key.get(key)
        if previous is None or _label_completeness(row) >= _label_completeness(previous):
            merged_by_key[key] = row
    return tuple(
        sorted(
            merged_by_key.values(),
            key=lambda row: (row.get("timestamp", ""), row.get("asset", ""), row.get("status", ""), row.get("side", "")),
        )
    )


def summarize_label_history(
    rows: tuple[dict[str, str], ...],
) -> tuple[DislocationLabelHistorySummary, ...]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row.get("asset", ""), row.get("status", ""), row.get("side", "")), []).append(row)
    summaries = tuple(
        _summarize_group(asset=key[0], status=key[1], side=key[2], rows=tuple(group_rows))
        for key, group_rows in grouped.items()
        if key[0] and key[1] and key[2]
    )
    return tuple(sorted(summaries, key=_summary_sort_key, reverse=True))


def write_label_history_csv(rows: tuple[dict[str, str], ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys()) if rows else _label_fieldnames()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_label_history_summary_csv(
    rows: tuple[DislocationLabelHistorySummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "asset",
                "status",
                "side",
                "observations",
                "covered_15m",
                "covered_1h",
                "covered_4h",
                "win_15m",
                "win_1h",
                "win_4h",
                "mean_net_15m_bps",
                "mean_net_1h_bps",
                "mean_net_4h_bps",
                "best_net_15m_bps",
                "best_net_1h_bps",
                "best_net_4h_bps",
                "history_action",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.status,
                    row.side,
                    row.observations,
                    row.covered_15m,
                    row.covered_1h,
                    row.covered_4h,
                    row.win_15m,
                    row.win_1h,
                    row.win_4h,
                    f"{row.mean_net_15m_bps:.8f}",
                    f"{row.mean_net_1h_bps:.8f}",
                    f"{row.mean_net_4h_bps:.8f}",
                    f"{row.best_net_15m_bps:.8f}",
                    f"{row.best_net_1h_bps:.8f}",
                    f"{row.best_net_4h_bps:.8f}",
                    row.history_action,
                )
            )
    return output_path


def write_label_history_summary_md(
    rows: tuple[DislocationLabelHistorySummary, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Dislocation Label History\n\n")
        handle.write(
            "This preserves dislocation forward labels across refreshed snapshots. "
            "It is a paper-label history, not a live fill record.\n\n"
        )
        handle.write(
            "| asset | status | side | action | obs | cov15 | win15 | mean15 | best15 | cov1h | win1h | mean1h |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.asset} | "
                f"{row.status} | "
                f"{row.side} | "
                f"{row.history_action} | "
                f"{row.observations} | "
                f"{row.covered_15m} | "
                f"{row.win_15m} | "
                f"{row.mean_net_15m_bps:.2f} | "
                f"{row.best_net_15m_bps:.2f} | "
                f"{row.covered_1h} | "
                f"{row.win_1h} | "
                f"{row.mean_net_1h_bps:.2f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows with repeated covered wins become stronger candidates for fresh paper probes. "
            "Rows with pending 1h/4h labels should be refreshed after enough time has elapsed.\n"
        )
    return output_path


def _summarize_group(
    *,
    asset: str,
    status: str,
    side: str,
    rows: tuple[dict[str, str], ...],
) -> DislocationLabelHistorySummary:
    net15 = tuple(_float(row.get("net_15m_bps")) for row in rows if row.get("net_15m_bps"))
    net1h = tuple(_float(row.get("net_1h_bps")) for row in rows if row.get("net_1h_bps"))
    net4h = tuple(_float(row.get("net_4h_bps")) for row in rows if row.get("net_4h_bps"))
    covered_15m = len(net15)
    covered_1h = len(net1h)
    covered_4h = len(net4h)
    win_15m = sum(1 for row in rows if row.get("outcome_15m") == "paper_15m_win")
    win_1h = sum(1 for row in rows if row.get("outcome_1h") == "paper_1h_win")
    win_4h = sum(1 for row in rows if row.get("outcome_4h") == "paper_4h_win")
    return DislocationLabelHistorySummary(
        asset=asset,
        status=status,
        side=side,
        observations=len(rows),
        covered_15m=covered_15m,
        covered_1h=covered_1h,
        covered_4h=covered_4h,
        win_15m=win_15m,
        win_1h=win_1h,
        win_4h=win_4h,
        mean_net_15m_bps=_mean(net15),
        mean_net_1h_bps=_mean(net1h),
        mean_net_4h_bps=_mean(net4h),
        best_net_15m_bps=max(net15) if net15 else 0.0,
        best_net_1h_bps=max(net1h) if net1h else 0.0,
        best_net_4h_bps=max(net4h) if net4h else 0.0,
        history_action=_history_action(
            covered_15m=covered_15m,
            covered_1h=covered_1h,
            win_15m=win_15m,
            win_1h=win_1h,
            mean_net_15m_bps=_mean(net15),
            mean_net_1h_bps=_mean(net1h),
        ),
    )


def _history_action(
    *,
    covered_15m: int,
    covered_1h: int,
    win_15m: int,
    win_1h: int,
    mean_net_15m_bps: float,
    mean_net_1h_bps: float,
) -> str:
    if covered_1h and win_1h and mean_net_1h_bps > 0.0:
        return "repeat_paper_probe_priority"
    if covered_15m and win_15m and mean_net_15m_bps > 0.0:
        return "wait_for_1h_or_repeat_label"
    if covered_15m:
        return "deprioritize_until_fresh_snapshot"
    return "pending_label"


def _summary_sort_key(row: DislocationLabelHistorySummary) -> tuple[int, int, float, int, float]:
    action_rank = {
        "repeat_paper_probe_priority": 4,
        "wait_for_1h_or_repeat_label": 3,
        "pending_label": 2,
        "deprioritize_until_fresh_snapshot": 1,
    }.get(row.history_action, 0)
    return (
        action_rank,
        row.win_1h,
        row.mean_net_1h_bps,
        row.win_15m,
        row.mean_net_15m_bps,
    )


def _label_completeness(row: dict[str, str]) -> tuple[int, int, int]:
    return (
        int(bool(row.get("net_4h_bps"))),
        int(bool(row.get("net_1h_bps"))),
        int(bool(row.get("net_15m_bps"))),
    )


def _label_fieldnames() -> tuple[str, ...]:
    return (
        "timestamp",
        "asset",
        "status",
        "side",
        "score",
        "direction",
        "annualized_funding",
        "impact_spread",
        "conservative_cost_bps",
        "raw_return_15m",
        "raw_return_1h",
        "raw_return_4h",
        "directional_return_15m",
        "directional_return_1h",
        "directional_return_4h",
        "funding_return_15m",
        "funding_return_1h",
        "funding_return_4h",
        "net_15m_bps",
        "net_1h_bps",
        "net_4h_bps",
        "outcome_15m",
        "outcome_1h",
        "outcome_4h",
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-label-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_forward_labels.csv",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_label_history.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_label_history_summary.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_label_history_summary.md",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    history = merge_label_history(current_label_path=args.current_label_path, history_path=args.history_path)
    summaries = summarize_label_history(history)
    write_label_history_csv(history, output_path=args.history_path)
    write_label_history_summary_csv(summaries, output_path=args.summary_output_path)
    write_label_history_summary_md(summaries, output_path=args.markdown_output_path, top=args.top)
    for row in summaries[: args.top]:
        print(
            row.asset,
            row.status,
            row.side,
            row.history_action,
            f"win15={row.win_15m}/{row.covered_15m}",
            f"win1h={row.win_1h}/{row.covered_1h}",
        )


if __name__ == "__main__":
    main()
