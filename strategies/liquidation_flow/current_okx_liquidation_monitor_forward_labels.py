from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from strategies.liquidation_flow.current_okx_liquidation_forward_labels import (
    LiquidationForwardLabel,
    build_liquidation_forward_labels,
    write_liquidation_forward_labels,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MonitorForwardLabelSummary:
    asset: str
    action: str
    observations: int
    coverage_15m: int
    hit_rate_15m: float | None
    mean_continuation_return_15m: float | None
    coverage_1h: int
    hit_rate_1h: float | None
    mean_continuation_return_1h: float | None


def build_monitor_forward_label_summary(
    labels: tuple[LiquidationForwardLabel, ...],
) -> tuple[MonitorForwardLabelSummary, ...]:
    keys = sorted({(label.asset, label.action) for label in labels})
    rows = tuple(
        _build_summary_row(
            asset=asset,
            action=action,
            labels=tuple(
                label
                for label in labels
                if label.asset == asset and label.action == action
            ),
        )
        for asset, action in keys
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.coverage_15m,
                row.mean_continuation_return_15m or -1.0,
                row.coverage_1h,
                row.mean_continuation_return_1h or -1.0,
            ),
            reverse=True,
        )
    )


def write_monitor_forward_label_summary(
    rows: tuple[MonitorForwardLabelSummary, ...],
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
                "observations",
                "coverage_15m",
                "hit_rate_15m",
                "mean_continuation_return_15m",
                "coverage_1h",
                "hit_rate_1h",
                "mean_continuation_return_1h",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.asset,
                    row.action,
                    row.observations,
                    row.coverage_15m,
                    "" if row.hit_rate_15m is None else f"{row.hit_rate_15m:.8f}",
                    (
                        ""
                        if row.mean_continuation_return_15m is None
                        else f"{row.mean_continuation_return_15m:.8f}"
                    ),
                    row.coverage_1h,
                    "" if row.hit_rate_1h is None else f"{row.hit_rate_1h:.8f}",
                    (
                        ""
                        if row.mean_continuation_return_1h is None
                        else f"{row.mean_continuation_return_1h:.8f}"
                    ),
                )
            )
    return output_path


def write_monitor_forward_label_summary_md(
    rows: tuple[MonitorForwardLabelSummary, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current OKX Liquidation Monitor Forward Label Summary\n\n")
        handle.write(
            "This labels repeated liquidation-monitor samples from each event "
            "timestamp. Positive continuation means price moved in the forced-flow "
            "direction over the horizon.\n\n"
        )
        handle.write(
            "| asset | action | obs | cov15 | hit15 | mean cont15 | cov1h | hit1h | mean cont1h |\n"
        )
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows[:top]:
            handle.write(
                "| "
                f"{row.asset} | "
                f"{row.action} | "
                f"{row.observations} | "
                f"{row.coverage_15m} | "
                f"{'' if row.hit_rate_15m is None else f'{row.hit_rate_15m:.4f}'} | "
                f"{'' if row.mean_continuation_return_15m is None else f'{row.mean_continuation_return_15m:.6f}'} | "
                f"{row.coverage_1h} | "
                f"{'' if row.hit_rate_1h is None else f'{row.hit_rate_1h:.4f}'} | "
                f"{'' if row.mean_continuation_return_1h is None else f'{row.mean_continuation_return_1h:.6f}'} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This is still event-label evidence, not a trading decision. It should "
            "be joined with depth, fees, funding, and venue availability before "
            "sizing a paper trade.\n"
        )
    return output_path


def _build_summary_row(
    *,
    asset: str,
    action: str,
    labels: tuple[LiquidationForwardLabel, ...],
) -> MonitorForwardLabelSummary:
    continuation_15m = tuple(
        label.continuation_return_15m
        for label in labels
        if label.continuation_return_15m is not None
    )
    continuation_1h = tuple(
        label.continuation_return_1h
        for label in labels
        if label.continuation_return_1h is not None
    )
    return MonitorForwardLabelSummary(
        asset=asset,
        action=action,
        observations=len(labels),
        coverage_15m=len(continuation_15m),
        hit_rate_15m=_hit_rate(continuation_15m),
        mean_continuation_return_15m=_mean(continuation_15m),
        coverage_1h=len(continuation_1h),
        hit_rate_1h=_hit_rate(continuation_1h),
        mean_continuation_return_1h=_mean(continuation_1h),
    )


def _hit_rate(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value > 0.0) / len(values)


def _mean(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_samples.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_forward_labels.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_forward_label_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_okx_liquidation_monitor_forward_label_summary.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    labels = build_liquidation_forward_labels(input_path=args.input_path)
    summary = build_monitor_forward_label_summary(labels)
    write_liquidation_forward_labels(labels, output_path=args.output_path)
    write_monitor_forward_label_summary(summary, output_path=args.summary_output_path)
    write_monitor_forward_label_summary_md(
        summary,
        output_path=args.md_output_path,
        top=args.top,
    )
    for row in summary[: args.top]:
        print(
            row.asset,
            row.action,
            f"cov15={row.coverage_15m}",
            "mean15="
            f"{'' if row.mean_continuation_return_15m is None else f'{row.mean_continuation_return_15m:.4f}'}",
            f"hit15={'' if row.hit_rate_15m is None else f'{row.hit_rate_15m:.2f}'}",
        )


if __name__ == "__main__":
    main()
