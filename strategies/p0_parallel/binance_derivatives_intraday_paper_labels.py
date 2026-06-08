from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
FEATURES = {
    "oi_value_change",
    "sum_top_long_short_ratio",
    "count_top_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
    "premium_close",
    "abs_premium_close",
}
ACTIONS = ("long_preferred", "short_opposite", "long_preferred_short_opposite")


@dataclass(frozen=True)
class PaperLabel:
    symbol: str
    feature: str
    action: str
    status: str
    bucket: str
    round_trip_cost_bps: float
    prior_trades: int
    recent_trades: int
    combined_trades: int
    prior_net_mean_1h: float
    recent_net_mean_1h: float
    combined_net_mean_1h: float
    prior_hit_rate: float
    recent_hit_rate: float
    combined_hit_rate: float
    score: float
    next_step: str


def build_intraday_paper_labels(
    *,
    repeat_path: Path,
    prior_labels_path: Path,
    recent_labels_path: Path,
    round_trip_cost_bps: float,
) -> tuple[PaperLabel, ...]:
    prior_labels = _labels_by_symbol(prior_labels_path)
    recent_labels = _labels_by_symbol(recent_labels_path)
    output: list[PaperLabel] = []
    for row in _read_rows(repeat_path):
        if row.get("status") not in {"intraday_repeat_priority", "intraday_repeat_watch"}:
            continue
        symbol = row.get("symbol", "")
        feature = row.get("feature", "")
        bucket = row.get("recent_bucket", "")
        if feature not in FEATURES or bucket not in {"high", "low"}:
            continue
        for action in ACTIONS:
            prior_returns = _action_returns(
                labels=prior_labels.get(symbol, ()),
                feature=feature,
                bucket=bucket,
                action=action,
                round_trip_cost_bps=round_trip_cost_bps,
            )
            recent_returns = _action_returns(
                labels=recent_labels.get(symbol, ()),
                feature=feature,
                bucket=bucket,
                action=action,
                round_trip_cost_bps=round_trip_cost_bps,
            )
            output.append(
                _paper_label(
                    symbol=symbol,
                    feature=feature,
                    action=action,
                    bucket=bucket,
                    round_trip_cost_bps=round_trip_cost_bps,
                    prior_returns=prior_returns,
                    recent_returns=recent_returns,
                )
            )
    return tuple(sorted(output, key=lambda label: label.score, reverse=True))


def write_intraday_paper_labels_csv(rows: tuple[PaperLabel, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "feature",
                "action",
                "status",
                "bucket",
                "round_trip_cost_bps",
                "prior_trades",
                "recent_trades",
                "combined_trades",
                "prior_net_mean_1h",
                "recent_net_mean_1h",
                "combined_net_mean_1h",
                "prior_hit_rate",
                "recent_hit_rate",
                "combined_hit_rate",
                "score",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.feature,
                    row.action,
                    row.status,
                    row.bucket,
                    f"{row.round_trip_cost_bps:.4f}",
                    row.prior_trades,
                    row.recent_trades,
                    row.combined_trades,
                    f"{row.prior_net_mean_1h:.12f}",
                    f"{row.recent_net_mean_1h:.12f}",
                    f"{row.combined_net_mean_1h:.12f}",
                    f"{row.prior_hit_rate:.8f}",
                    f"{row.recent_hit_rate:.8f}",
                    f"{row.combined_hit_rate:.8f}",
                    f"{row.score:.8f}",
                    row.next_step,
                )
            )
    return output_path


def write_intraday_paper_labels_md(
    rows: tuple[PaperLabel, ...],
    *,
    output_path: Path,
    top: int = 60,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Binance Derivatives Intraday Paper Labels\n\n")
        handle.write(
            "This applies a rough round-trip cost to repeated 5m-to-1h derivatives feature buckets. "
            "It tests long preferred bucket, short opposite bucket, and the combined action. "
            "It is still a paper label screen, not a live order plan.\n\n"
        )
        handle.write(
            "| symbol | feature | action | status | bucket | cost bps | prior trades | recent trades | prior net 1h | recent net 1h | combined net 1h | combined hit | score | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.feature} | {row.action} | {row.status} | "
                f"{row.bucket} | {row.round_trip_cost_bps:.2f} | "
                f"{row.prior_trades} | {row.recent_trades} | "
                f"{row.prior_net_mean_1h:.6f} | {row.recent_net_mean_1h:.6f} | "
                f"{row.combined_net_mean_1h:.6f} | {row.combined_hit_rate:.4f} | "
                f"{row.score:.4f} | {row.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`paper_intraday_cost_supported` rows keep positive net 1h means in both non-overlapping windows after rough cost. "
            "`paper_intraday_recent_only` rows fail the prior window and need more history before promotion.\n"
        )
    return output_path


def _paper_label(
    *,
    symbol: str,
    feature: str,
    action: str,
    bucket: str,
    round_trip_cost_bps: float,
    prior_returns: tuple[float, ...],
    recent_returns: tuple[float, ...],
) -> PaperLabel:
    combined_returns = prior_returns + recent_returns
    prior_mean = _mean(prior_returns)
    recent_mean = _mean(recent_returns)
    combined_mean = _mean(combined_returns)
    prior_hit = _hit_rate(prior_returns)
    recent_hit = _hit_rate(recent_returns)
    combined_hit = _hit_rate(combined_returns)
    status = _status(
        prior_mean=prior_mean,
        recent_mean=recent_mean,
        combined_mean=combined_mean,
        prior_hit=prior_hit,
        recent_hit=recent_hit,
        combined_hit=combined_hit,
        combined_trades=len(combined_returns),
    )
    return PaperLabel(
        symbol=symbol,
        feature=feature,
        action=action,
        status=status,
        bucket=bucket,
        round_trip_cost_bps=round_trip_cost_bps,
        prior_trades=len(prior_returns),
        recent_trades=len(recent_returns),
        combined_trades=len(combined_returns),
        prior_net_mean_1h=prior_mean,
        recent_net_mean_1h=recent_mean,
        combined_net_mean_1h=combined_mean,
        prior_hit_rate=prior_hit,
        recent_hit_rate=recent_hit,
        combined_hit_rate=combined_hit,
        score=_score(
            prior_mean=prior_mean,
            recent_mean=recent_mean,
            combined_mean=combined_mean,
            combined_hit=combined_hit,
            combined_trades=len(combined_returns),
            status=status,
        ),
        next_step=_next_step(symbol=symbol, feature=feature, action=action, status=status),
    )


def _action_returns(
    *,
    labels: tuple[dict[str, float], ...],
    feature: str,
    bucket: str,
    action: str,
    round_trip_cost_bps: float,
) -> tuple[float, ...]:
    if not labels:
        return ()
    values = sorted(label[feature] for label in labels)
    threshold = values[int(len(values) * (0.25 if bucket == "low" else 0.75))]
    cost = round_trip_cost_bps / 10_000.0
    returns: list[float] = []
    for label in labels:
        feature_value = label[feature]
        next_return = label["next_1h_return"]
        if action in {"long_preferred", "long_preferred_short_opposite"} and _in_bucket(
            feature_value=feature_value,
            threshold=threshold,
            bucket=bucket,
        ):
            returns.append(next_return - cost)
        if action in {"short_opposite", "long_preferred_short_opposite"} and not _in_bucket(
            feature_value=feature_value,
            threshold=threshold,
            bucket=bucket,
        ):
            returns.append(-next_return - cost)
    return tuple(returns)


def _in_bucket(*, feature_value: float, threshold: float, bucket: str) -> bool:
    if bucket == "low":
        return feature_value <= threshold
    return feature_value >= threshold


def _status(
    *,
    prior_mean: float,
    recent_mean: float,
    combined_mean: float,
    prior_hit: float,
    recent_hit: float,
    combined_hit: float,
    combined_trades: int,
) -> str:
    if combined_trades < 500:
        return "paper_intraday_thin"
    if prior_mean > 0.0 and recent_mean > 0.0 and min(prior_hit, recent_hit) >= 0.50:
        return "paper_intraday_cost_supported"
    if combined_mean > 0.0 and combined_hit >= 0.50 and recent_mean > 0.0:
        return "paper_intraday_recent_only"
    if combined_mean > 0.0:
        return "paper_intraday_positive_mean_watch"
    return "paper_intraday_cost_rejected"


def _score(
    *,
    prior_mean: float,
    recent_mean: float,
    combined_mean: float,
    combined_hit: float,
    combined_trades: int,
    status: str,
) -> float:
    base = combined_mean * 100_000.0 + combined_hit * 40.0 + min(combined_trades / 100.0, 40.0)
    if status == "paper_intraday_cost_supported":
        base += 120.0
    elif status == "paper_intraday_recent_only":
        base += 40.0
    elif status == "paper_intraday_cost_rejected":
        base -= 80.0
    repeat_bonus = 40.0 if prior_mean > 0.0 and recent_mean > 0.0 else 0.0
    return base + repeat_bonus


def _next_step(*, symbol: str, feature: str, action: str, status: str) -> str:
    if status == "paper_intraday_cost_supported":
        return f"paper-check {symbol} {feature} {action} with live spread, funding timestamp, fill delay, and stop rules"
    if status == "paper_intraday_recent_only":
        return f"extend {symbol} {feature} {action} to another non-overlapping window before promotion"
    return f"do not promote {symbol} {feature} {action} without stronger cost-aware repeat evidence"


def _labels_by_symbol(path: Path) -> dict[str, tuple[dict[str, float], ...]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in _read_rows(path):
        symbol = row.get("symbol", "")
        if not symbol:
            continue
        try:
            grouped.setdefault(symbol, []).append(
                {
                    "next_1h_return": float(row["next_1h_return"]),
                    **{feature: float(row[feature]) for feature in FEATURES},
                }
            )
        except (KeyError, ValueError):
            continue
    return {symbol: tuple(rows) for symbol, rows in grouped.items()}


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _hit_rate(values: tuple[float, ...]) -> float:
    return mean(1.0 if value > 0.0 else 0.0 for value in values) if values else 0.0


def _mean(values: tuple[float, ...]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeat-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_repeat_compare.csv",
    )
    parser.add_argument(
        "--prior-labels-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_prior_feature_labels.csv",
    )
    parser.add_argument(
        "--recent-labels-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_feature_labels.csv",
    )
    parser.add_argument("--round-trip-cost-bps", type=float, default=8.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_paper_labels.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_paper_labels.md",
    )
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    rows = build_intraday_paper_labels(
        repeat_path=args.repeat_path,
        prior_labels_path=args.prior_labels_path,
        recent_labels_path=args.recent_labels_path,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    write_intraday_paper_labels_csv(rows, output_path=args.output_path)
    write_intraday_paper_labels_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.feature, row.action, row.status, f"score={row.score:.4f}")


if __name__ == "__main__":
    main()
