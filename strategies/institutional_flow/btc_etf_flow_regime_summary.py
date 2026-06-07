from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BtcEtfFlowRegimeSummary:
    group_key: str
    observations: int
    mean_flow_btc: float
    mean_rolling_5d_flow_btc: float
    mean_directional_1d: float
    mean_directional_3d: float
    mean_directional_5d: float
    hit_rate_5d: float
    action: str


def build_regime_summaries(*, labels_path: Path) -> tuple[BtcEtfFlowRegimeSummary, ...]:
    rows = tuple(row for row in _read_rows(labels_path) if row.get("directional_return_5d"))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["action"], []).append(row)
        grouped.setdefault(_flow_size_group(row), []).append(row)
        grouped.setdefault(_rolling_size_group(row), []).append(row)
    summaries = tuple(_summarize_group(group_key=key, rows=tuple(value)) for key, value in grouped.items())
    return tuple(sorted(summaries, key=lambda row: (row.mean_directional_5d, row.hit_rate_5d), reverse=True))


def write_summaries_csv(
    summaries: tuple[BtcEtfFlowRegimeSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "group_key",
                "observations",
                "mean_flow_btc",
                "mean_rolling_5d_flow_btc",
                "mean_directional_1d",
                "mean_directional_3d",
                "mean_directional_5d",
                "hit_rate_5d",
                "action",
            )
        )
        for summary in summaries:
            writer.writerow(
                (
                    summary.group_key,
                    summary.observations,
                    f"{summary.mean_flow_btc:.8f}",
                    f"{summary.mean_rolling_5d_flow_btc:.8f}",
                    f"{summary.mean_directional_1d:.8f}",
                    f"{summary.mean_directional_3d:.8f}",
                    f"{summary.mean_directional_5d:.8f}",
                    f"{summary.hit_rate_5d:.8f}",
                    summary.action,
                )
            )
    return output_path


def write_summaries_md(
    summaries: tuple[BtcEtfFlowRegimeSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# BTC ETF Flow Regime Summary\n\n")
        handle.write(
            "This splits BTC ETF flow forward labels by action and flow-size regimes. It is not net PnL.\n\n"
        )
        handle.write(
            "| group | obs | mean flow BTC | mean 5d flow BTC | mean dir 1d | mean dir 3d | mean dir 5d | hit 5d | action |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for summary in summaries:
            handle.write(
                f"| {summary.group_key} | {summary.observations} | "
                f"{summary.mean_flow_btc:.2f} | {summary.mean_rolling_5d_flow_btc:.2f} | "
                f"{summary.mean_directional_1d:.8f} | {summary.mean_directional_3d:.8f} | "
                f"{summary.mean_directional_5d:.8f} | {summary.hit_rate_5d:.4f} | "
                f"{summary.action} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The useful question is not whether ETF flow is universally predictive, but which flow regime survives leakage-safe forward labeling.\n"
        )
    return output_path


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _flow_size_group(row: dict[str, str]) -> str:
    flow = float(row["flow_btc"])
    if flow >= 5_000.0:
        return "large_daily_inflow"
    if flow <= -5_000.0:
        return "large_daily_outflow"
    return "small_daily_flow"


def _rolling_size_group(row: dict[str, str]) -> str:
    flow = float(row["rolling_5d_flow_btc"])
    if flow >= 15_000.0:
        return "large_5d_inflow"
    if flow <= -15_000.0:
        return "large_5d_outflow"
    return "mixed_5d_flow"


def _summarize_group(
    *,
    group_key: str,
    rows: tuple[dict[str, str], ...],
) -> BtcEtfFlowRegimeSummary:
    dir_1d = tuple(float(row["directional_return_1d"]) for row in rows if row.get("directional_return_1d"))
    dir_3d = tuple(float(row["directional_return_3d"]) for row in rows if row.get("directional_return_3d"))
    dir_5d = tuple(float(row["directional_return_5d"]) for row in rows if row.get("directional_return_5d"))
    summary = BtcEtfFlowRegimeSummary(
        group_key=group_key,
        observations=len(rows),
        mean_flow_btc=_mean(tuple(float(row["flow_btc"]) for row in rows)),
        mean_rolling_5d_flow_btc=_mean(tuple(float(row["rolling_5d_flow_btc"]) for row in rows)),
        mean_directional_1d=_mean(dir_1d),
        mean_directional_3d=_mean(dir_3d),
        mean_directional_5d=_mean(dir_5d),
        hit_rate_5d=_hit_rate(dir_5d),
        action="",
    )
    return BtcEtfFlowRegimeSummary(
        **{
            **summary.__dict__,
            "action": _action_for_summary(summary),
        }
    )


def _action_for_summary(summary: BtcEtfFlowRegimeSummary) -> str:
    if (
        summary.observations >= 30
        and summary.mean_directional_5d >= 0.01
        and summary.hit_rate_5d >= 0.55
    ):
        return "regime_candidate"
    if summary.observations >= 30 and summary.mean_directional_5d > 0.0:
        return "regime_watch"
    return "weak_or_insufficient"


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _hit_rate(values: tuple[float, ...]) -> float:
    return sum(1.0 for value in values if value > 0.0) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=ROOT / "btc_etf_flow_forward_labels.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_regime_summary.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "btc_etf_flow_regime_summary.md",
    )
    args = parser.parse_args()

    summaries = build_regime_summaries(labels_path=args.labels_path)
    write_summaries_csv(summaries, output_path=args.output_path)
    write_summaries_md(summaries, output_path=args.markdown_output_path)
    for summary in summaries[:10]:
        print(
            summary.group_key,
            f"obs={summary.observations}",
            f"mean5={summary.mean_directional_5d:.8f}",
            f"hit5={summary.hit_rate_5d:.4f}",
            summary.action,
        )


if __name__ == "__main__":
    main()
