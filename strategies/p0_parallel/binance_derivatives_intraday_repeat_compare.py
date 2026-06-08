from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class IntradayRepeatComparison:
    symbol: str
    feature: str
    status: str
    prior_bucket: str
    recent_bucket: str
    prior_score: float
    recent_score: float
    combined_score: float
    prior_mean_edge: float
    recent_mean_edge: float
    prior_hit_edge: float
    recent_hit_edge: float
    next_step: str


def build_intraday_repeat_comparisons(
    *,
    prior_path: Path,
    recent_path: Path,
) -> tuple[IntradayRepeatComparison, ...]:
    prior_rows = {
        (row.get("symbol", ""), row.get("feature", "")): row
        for row in _read_rows(prior_path)
        if row.get("symbol") and row.get("feature")
    }
    recent_rows = {
        (row.get("symbol", ""), row.get("feature", "")): row
        for row in _read_rows(recent_path)
        if row.get("symbol") and row.get("feature")
    }
    comparisons = [
        _comparison_for_key(
            symbol=symbol,
            feature=feature,
            prior=prior_rows.get((symbol, feature), {}),
            recent=recent_rows.get((symbol, feature), {}),
        )
        for symbol, feature in sorted(set(prior_rows) | set(recent_rows))
    ]
    return tuple(sorted(comparisons, key=lambda row: row.combined_score, reverse=True))


def write_intraday_repeat_comparisons_csv(
    rows: tuple[IntradayRepeatComparison, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "symbol",
                "feature",
                "status",
                "prior_bucket",
                "recent_bucket",
                "prior_score",
                "recent_score",
                "combined_score",
                "prior_mean_edge",
                "recent_mean_edge",
                "prior_hit_edge",
                "recent_hit_edge",
                "next_step",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.symbol,
                    row.feature,
                    row.status,
                    row.prior_bucket,
                    row.recent_bucket,
                    f"{row.prior_score:.8f}",
                    f"{row.recent_score:.8f}",
                    f"{row.combined_score:.8f}",
                    f"{row.prior_mean_edge:.12f}",
                    f"{row.recent_mean_edge:.12f}",
                    f"{row.prior_hit_edge:.8f}",
                    f"{row.recent_hit_edge:.8f}",
                    row.next_step,
                )
            )
    return output_path


def write_intraday_repeat_comparisons_md(
    rows: tuple[IntradayRepeatComparison, ...],
    *,
    output_path: Path,
    top: int = 60,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Binance Derivatives Intraday Repeat Compare\n\n")
        handle.write(
            "This compares a prior non-overlapping 5m-to-1h label window against the recent window. "
            "Rows with the same preferred bucket across windows are repeat candidates, not trade instructions.\n\n"
        )
        handle.write(
            "| symbol | feature | status | prior bucket | recent bucket | prior score | recent score | combined score | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.feature} | {row.status} | "
                f"{row.prior_bucket} | {row.recent_bucket} | "
                f"{row.prior_score:.4f} | {row.recent_score:.4f} | "
                f"{row.combined_score:.4f} | {row.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`intraday_repeat_priority` rows repeated the same symbol-feature bucket across non-overlapping windows. "
            "`intraday_recent_only_priority` rows are current but not repeated. "
            "`intraday_bucket_shift` rows changed direction and should not be promoted without a regime explanation.\n"
        )
    return output_path


def _comparison_for_key(
    *,
    symbol: str,
    feature: str,
    prior: dict[str, str],
    recent: dict[str, str],
) -> IntradayRepeatComparison:
    prior_bucket = prior.get("preferred_bucket", "")
    recent_bucket = recent.get("preferred_bucket", "")
    prior_score = _float(prior.get("edge_score"))
    recent_score = _float(recent.get("edge_score"))
    prior_mean_edge = _mean_edge(prior)
    recent_mean_edge = _mean_edge(recent)
    prior_hit_edge = _hit_edge(prior)
    recent_hit_edge = _hit_edge(recent)
    status = _status(
        prior_bucket=prior_bucket,
        recent_bucket=recent_bucket,
        prior_score=prior_score,
        recent_score=recent_score,
        prior_mean_edge=prior_mean_edge,
        recent_mean_edge=recent_mean_edge,
    )
    combined_score = _combined_score(
        status=status,
        prior_score=prior_score,
        recent_score=recent_score,
        prior_mean_edge=prior_mean_edge,
        recent_mean_edge=recent_mean_edge,
        prior_hit_edge=prior_hit_edge,
        recent_hit_edge=recent_hit_edge,
    )
    return IntradayRepeatComparison(
        symbol=symbol,
        feature=feature,
        status=status,
        prior_bucket=prior_bucket,
        recent_bucket=recent_bucket,
        prior_score=prior_score,
        recent_score=recent_score,
        combined_score=combined_score,
        prior_mean_edge=prior_mean_edge,
        recent_mean_edge=recent_mean_edge,
        prior_hit_edge=prior_hit_edge,
        recent_hit_edge=recent_hit_edge,
        next_step=_next_step(symbol=symbol, feature=feature, status=status),
    )


def _status(
    *,
    prior_bucket: str,
    recent_bucket: str,
    prior_score: float,
    recent_score: float,
    prior_mean_edge: float,
    recent_mean_edge: float,
) -> str:
    same_bucket = prior_bucket in {"high", "low"} and prior_bucket == recent_bucket
    same_mean_direction = prior_mean_edge * recent_mean_edge > 0.0
    if same_bucket and same_mean_direction and prior_score >= 140.0 and recent_score >= 140.0:
        return "intraday_repeat_priority"
    if same_bucket and same_mean_direction and min(prior_score, recent_score) >= 80.0:
        return "intraday_repeat_watch"
    if recent_score >= 140.0 and not prior_score:
        return "intraday_recent_only_priority"
    if recent_score >= 140.0 and prior_bucket != recent_bucket:
        return "intraday_bucket_shift"
    if recent_score >= 80.0:
        return "intraday_recent_watch"
    return "weak_intraday_repeat_context"


def _combined_score(
    *,
    status: str,
    prior_score: float,
    recent_score: float,
    prior_mean_edge: float,
    recent_mean_edge: float,
    prior_hit_edge: float,
    recent_hit_edge: float,
) -> float:
    base = min(prior_score, recent_score) + ((prior_score + recent_score) * 0.20)
    if status == "intraday_repeat_priority":
        base += 120.0
    elif status == "intraday_repeat_watch":
        base += 70.0
    elif status == "intraday_bucket_shift":
        base -= 80.0
    direction_bonus = 40.0 if prior_mean_edge * recent_mean_edge > 0.0 else 0.0
    hit_bonus = 20.0 if prior_hit_edge * recent_hit_edge > 0.0 else 0.0
    return base + direction_bonus + hit_bonus


def _next_step(*, symbol: str, feature: str, status: str) -> str:
    if status in {"intraday_repeat_priority", "intraday_repeat_watch"}:
        return (
            f"run {symbol} {feature} intraday paper label with fees, spread, funding PnL, "
            "fill assumptions, and stop behavior"
        )
    if status == "intraday_bucket_shift":
        return f"explain {symbol} {feature} bucket shift by regime before any promotion"
    return f"keep {symbol} {feature} as context until another non-overlapping window repeats"


def _mean_edge(row: dict[str, str]) -> float:
    return _float(row.get("high_bucket_mean_next_1h_return")) - _float(row.get("low_bucket_mean_next_1h_return"))


def _hit_edge(row: dict[str, str]) -> float:
    return _float(row.get("high_bucket_hit_rate")) - _float(row.get("low_bucket_hit_rate"))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prior-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_prior_feature_candidates.csv",
    )
    parser.add_argument(
        "--recent-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_feature_candidates.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_repeat_compare.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "binance_derivatives_intraday_repeat_compare.md",
    )
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    rows = build_intraday_repeat_comparisons(prior_path=args.prior_path, recent_path=args.recent_path)
    write_intraday_repeat_comparisons_csv(rows, output_path=args.output_path)
    write_intraday_repeat_comparisons_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(row.symbol, row.feature, row.status, f"combined={row.combined_score:.4f}")


if __name__ == "__main__":
    main()
