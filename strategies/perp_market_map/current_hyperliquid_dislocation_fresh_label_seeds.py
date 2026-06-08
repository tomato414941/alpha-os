from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build_fresh_label_seeds(
    *,
    queue_path: Path = ROOT / "current_hyperliquid_dislocation_repeat_label_queue.csv",
    candidate_path: Path = ROOT / "current_hyperliquid_dislocation_candidates.csv",
) -> tuple[dict[str, str], ...]:
    fresh_keys = {
        (row.get("asset", ""), row.get("status", ""), row.get("side", ""))
        for row in _read_rows(queue_path)
        if row.get("queue_action") == "fresh_forward_label_candidate"
    }
    candidates = (
        row
        for row in _read_rows(candidate_path)
        if (row.get("asset", ""), row.get("status", ""), row.get("side", "")) in fresh_keys
    )
    return tuple(sorted(candidates, key=lambda row: _float(row.get("score")), reverse=True))


def write_fresh_label_seeds_csv(rows: tuple[dict[str, str], ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys()) if rows else _candidate_fieldnames()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_fresh_label_seeds_md(rows: tuple[dict[str, str], ...], *, output_path: Path, top: int = 30) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Dislocation Fresh Label Seeds\n\n")
        handle.write(
            "These are repeated current candidates that need a fresh forward-label "
            "window. They are written as candidate rows so the existing forward-label "
            "runner can label them without overwriting the main label file.\n\n"
        )
        handle.write(f"- seeds: `{len(rows)}`\n\n")
        handle.write("| asset | status | side | score | ret24 | funding ann | impact | next step |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.get('asset', '')} | "
                f"{row.get('status', '')} | "
                f"{row.get('side', '')} | "
                f"{_float(row.get('score')):.4f} | "
                f"{_float(row.get('return_24h')):.4f} | "
                f"{_float(row.get('annualized_funding')):.4f} | "
                f"{_float(row.get('impact_spread')):.6f} | "
                f"{row.get('next_step', '')} |\n"
            )
    return output_path


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _candidate_fieldnames() -> tuple[str, ...]:
    return (
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


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_repeat_label_queue.csv",
    )
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_candidates.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_fresh_label_seeds.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_fresh_label_seeds.md",
    )
    args = parser.parse_args()

    rows = build_fresh_label_seeds(queue_path=args.queue_path, candidate_path=args.candidate_path)
    write_fresh_label_seeds_csv(rows, output_path=args.output_path)
    write_fresh_label_seeds_md(rows, output_path=args.md_output_path)
    for row in rows:
        print(
            row.get("asset", ""),
            row.get("status", ""),
            row.get("side", ""),
            f"score={_float(row.get('score')):.4f}",
        )


if __name__ == "__main__":
    main()
