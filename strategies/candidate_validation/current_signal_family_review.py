from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class SignalFamilyRow:
    family: str
    observations: int
    coverage_15m: int
    mean_label_15m: float | None
    hit_rate_15m: float | None
    max_label_15m: float | None
    min_label_15m: float | None
    support_score: float
    note: str


def build_signal_family_rows() -> tuple[SignalFamilyRow, ...]:
    groups: dict[str, list[float | None]] = {}
    _add_hl_candidate_groups(groups)
    _add_okx_pressure_groups(groups)
    _add_liquidation_groups(groups)
    _add_l2_imbalance_groups(groups)
    _add_sector_rotation_groups(groups)
    rows = tuple(_summarize_family(family=family, labels=tuple(labels)) for family, labels in groups.items())
    return tuple(sorted(rows, key=lambda row: row.support_score, reverse=True))


def write_signal_family_rows(
    rows: tuple[SignalFamilyRow, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "family",
                "observations",
                "coverage_15m",
                "mean_label_15m",
                "hit_rate_15m",
                "max_label_15m",
                "min_label_15m",
                "support_score",
                "note",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.family,
                    row.observations,
                    row.coverage_15m,
                    "" if row.mean_label_15m is None else f"{row.mean_label_15m:.8f}",
                    "" if row.hit_rate_15m is None else f"{row.hit_rate_15m:.8f}",
                    "" if row.max_label_15m is None else f"{row.max_label_15m:.8f}",
                    "" if row.min_label_15m is None else f"{row.min_label_15m:.8f}",
                    f"{row.support_score:.8f}",
                    row.note,
                )
            )
    return output_path


def write_signal_family_md(
    rows: tuple[SignalFamilyRow, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Signal Family Review\n\n")
        handle.write(
            "This aggregates short-horizon labels by signal family. It asks which "
            "kind of signal is currently showing support, not only which asset is "
            "on top.\n\n"
        )
        handle.write(
            "| family | obs | cov15 | mean15 | hit15 | max15 | min15 | score | note |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.family} | "
                f"{row.observations} | "
                f"{row.coverage_15m} | "
                f"{'' if row.mean_label_15m is None else f'{row.mean_label_15m:.6f}'} | "
                f"{'' if row.hit_rate_15m is None else f'{row.hit_rate_15m:.6f}'} | "
                f"{'' if row.max_label_15m is None else f'{row.max_label_15m:.6f}'} | "
                f"{'' if row.min_label_15m is None else f'{row.min_label_15m:.6f}'} | "
                f"{row.support_score:.6f} | "
                f"{row.note} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This is a small live-label summary. It is useful for prioritizing "
            "which signal families deserve repeated sampling, but it is not a "
            "backtest or execution-ready PnL estimate.\n"
        )
    return output_path


def _add_hl_candidate_groups(groups: dict[str, list[float | None]]) -> None:
    path = ROOT / "current_hl_signal_forward_label_summary.csv"
    for row in _read_rows(path):
        family = f"hl_candidate:{row['source']}:{row['action']}"
        labels = groups.setdefault(family, [])
        mean15 = _float_or_none(row.get("mean_return_15m", ""))
        coverage = int(row.get("coverage_15m") or "0")
        observations = int(row.get("observations") or "0")
        labels.extend([mean15] * max(coverage, 0))
        labels.extend([None] * max(observations - coverage, 0))


def _add_okx_pressure_groups(groups: dict[str, list[float | None]]) -> None:
    path = STRATEGIES_ROOT / "perp_market_map" / "current_okx_perp_pressure_forward_labels.csv"
    for row in _read_rows(path):
        family = f"okx_pressure:{row['action']}"
        groups.setdefault(family, []).append(
            _float_or_none(row.get("directional_return_15m", ""))
        )


def _add_liquidation_groups(groups: dict[str, list[float | None]]) -> None:
    path = STRATEGIES_ROOT / "liquidation_flow" / "current_okx_liquidation_forward_labels.csv"
    for row in _read_rows(path):
        family = f"okx_liquidation:{row['action']}"
        groups.setdefault(family, []).append(
            _float_or_none(row.get("continuation_return_15m", ""))
        )


def _add_l2_imbalance_groups(groups: dict[str, list[float | None]]) -> None:
    path = STRATEGIES_ROOT / "market_making" / "current_l2_imbalance_forward_labels.csv"
    for row in _read_rows(path):
        groups.setdefault("l2_imbalance:visible_book_imbalance", []).append(
            _float_or_none(row.get("directional_return_15m", ""))
        )


def _add_sector_rotation_groups(groups: dict[str, list[float | None]]) -> None:
    path = STRATEGIES_ROOT / "sector_rotation" / "current_category_tradable_forward_labels.csv"
    for row in _read_rows(path):
        if row.get("label_status") != "tradable_labeled":
            continue
        family = f"sector_rotation:{row['category_action']}"
        groups.setdefault(family, []).append(
            _float_or_none(row.get("directional_return_15m", ""))
        )


def _summarize_family(*, family: str, labels: tuple[float | None, ...]) -> SignalFamilyRow:
    covered = tuple(label for label in labels if label is not None)
    mean_label = _mean_or_none(covered)
    hit_rate = _hit_rate_or_none(covered)
    support_score = _support_score(covered=covered, mean_label=mean_label, hit_rate=hit_rate)
    return SignalFamilyRow(
        family=family,
        observations=len(labels),
        coverage_15m=len(covered),
        mean_label_15m=mean_label,
        hit_rate_15m=hit_rate,
        max_label_15m=max(covered) if covered else None,
        min_label_15m=min(covered) if covered else None,
        support_score=support_score,
        note=_note(coverage=len(covered), mean_label=mean_label, hit_rate=hit_rate),
    )


def _support_score(
    *,
    covered: tuple[float, ...],
    mean_label: float | None,
    hit_rate: float | None,
) -> float:
    if mean_label is None or hit_rate is None:
        return 0.0
    coverage_score = min(len(covered), 20) / 20.0
    return coverage_score * max(hit_rate - 0.5, 0.0) * max(mean_label, 0.0) * 1000.0


def _note(*, coverage: int, mean_label: float | None, hit_rate: float | None) -> str:
    if coverage == 0:
        return "waiting for elapsed labels"
    if mean_label is not None and mean_label > 0.0 and hit_rate is not None and hit_rate >= 0.6:
        return "supported by first labels"
    if mean_label is not None and mean_label > 0.0:
        return "positive mean but weak hit rate"
    return "not supported by first labels"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def _hit_rate_or_none(values: tuple[float, ...]) -> float | None:
    return sum(value > 0.0 for value in values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_signal_family_review.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_signal_family_review.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = build_signal_family_rows()
    write_signal_family_rows(rows, output_path=args.output_path)
    write_signal_family_md(rows, output_path=args.md_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.family,
            f"cov15={row.coverage_15m}",
            f"mean15={'' if row.mean_label_15m is None else f'{row.mean_label_15m:.4f}'}",
            f"hit15={'' if row.hit_rate_15m is None else f'{row.hit_rate_15m:.2f}'}",
            row.note,
        )


if __name__ == "__main__":
    main()
