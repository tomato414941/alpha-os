from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
FEATURES = (
    "mean_premium_close",
    "max_abs_premium_close",
    "mean_funding_rate",
    "sum_funding_rate",
    "oi_value_change",
    "mean_count_top_long_short_ratio",
    "mean_sum_top_long_short_ratio",
    "mean_count_long_short_ratio",
    "mean_sum_taker_long_short_vol_ratio",
)


@dataclass(frozen=True)
class SymbolFeatureCandidate:
    symbol: str
    feature: str
    status: str
    preferred_bucket: str
    observations: int
    low_bucket_mean_next_return: float
    low_bucket_hit_rate: float
    high_bucket_mean_next_return: float
    high_bucket_hit_rate: float
    correlation_to_next_return: float
    edge_score: float
    next_step: str


def build_symbol_feature_candidates(
    *,
    history_path: Path,
) -> tuple[SymbolFeatureCandidate, ...]:
    rows = tuple(row for row in _read_rows(history_path) if _float(row.get("next_return")) != 0.0)
    by_symbol: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_symbol.setdefault(row.get("symbol", ""), []).append(row)

    candidates: list[SymbolFeatureCandidate] = []
    for symbol, symbol_rows in sorted(by_symbol.items()):
        if not symbol or len(symbol_rows) < 20:
            continue
        for feature in FEATURES:
            candidates.append(_candidate_for_feature(symbol=symbol, feature=feature, rows=tuple(symbol_rows)))
    return tuple(sorted(candidates, key=lambda candidate: candidate.edge_score, reverse=True))


def write_symbol_feature_candidates_csv(
    candidates: tuple[SymbolFeatureCandidate, ...],
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
                "preferred_bucket",
                "observations",
                "low_bucket_mean_next_return",
                "low_bucket_hit_rate",
                "high_bucket_mean_next_return",
                "high_bucket_hit_rate",
                "correlation_to_next_return",
                "edge_score",
                "next_step",
            )
        )
        for candidate in candidates:
            writer.writerow(
                (
                    candidate.symbol,
                    candidate.feature,
                    candidate.status,
                    candidate.preferred_bucket,
                    candidate.observations,
                    f"{candidate.low_bucket_mean_next_return:.12f}",
                    f"{candidate.low_bucket_hit_rate:.8f}",
                    f"{candidate.high_bucket_mean_next_return:.12f}",
                    f"{candidate.high_bucket_hit_rate:.8f}",
                    f"{candidate.correlation_to_next_return:.8f}",
                    f"{candidate.edge_score:.8f}",
                    candidate.next_step,
                )
            )
    return output_path


def write_symbol_feature_candidates_md(
    candidates: tuple[SymbolFeatureCandidate, ...],
    *,
    output_path: Path,
    top: int = 40,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Binance Derivatives Symbol Feature Candidates\n\n")
        handle.write(
            "This screen looks for symbol-specific Binance USD-M derivatives features that had "
            "different next-day return behavior in the current historical panel. It is a research "
            "queue, not a trade list.\n\n"
        )
        handle.write(
            "| symbol | feature | status | bucket | observations | low mean | low hit | high mean | high hit | corr | score | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for candidate in candidates[:top]:
            handle.write(
                f"| {candidate.symbol} | {candidate.feature} | {candidate.status} | "
                f"{candidate.preferred_bucket} | {candidate.observations} | "
                f"{candidate.low_bucket_mean_next_return:.6f} | "
                f"{candidate.low_bucket_hit_rate:.4f} | "
                f"{candidate.high_bucket_mean_next_return:.6f} | "
                f"{candidate.high_bucket_hit_rate:.4f} | "
                f"{candidate.correlation_to_next_return:.4f} | "
                f"{candidate.edge_score:.4f} | {candidate.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Rows with high scores are places where a feature behaved differently for one symbol. "
            "The useful next step is recent-window reruns and cost-aware execution checks, not direct promotion.\n"
        )
    return output_path


def _candidate_for_feature(
    *,
    symbol: str,
    feature: str,
    rows: tuple[dict[str, str], ...],
) -> SymbolFeatureCandidate:
    sorted_values = sorted(_float(row.get(feature)) for row in rows)
    low_threshold = sorted_values[int(len(sorted_values) * 0.25)]
    high_threshold = sorted_values[int(len(sorted_values) * 0.75)]
    low_returns = tuple(_float(row.get("next_return")) for row in rows if _float(row.get(feature)) <= low_threshold)
    high_returns = tuple(_float(row.get("next_return")) for row in rows if _float(row.get(feature)) >= high_threshold)
    low_mean = _mean(low_returns)
    high_mean = _mean(high_returns)
    low_hit = _hit_rate(low_returns)
    high_hit = _hit_rate(high_returns)
    correlation = _correlation(
        tuple(_float(row.get(feature)) for row in rows),
        tuple(_float(row.get("next_return")) for row in rows),
    )
    preferred_bucket = _preferred_bucket(low_mean=low_mean, low_hit=low_hit, high_mean=high_mean, high_hit=high_hit)
    edge_score = _edge_score(
        low_mean=low_mean,
        low_hit=low_hit,
        high_mean=high_mean,
        high_hit=high_hit,
        correlation=correlation,
        observations=len(rows),
    )
    return SymbolFeatureCandidate(
        symbol=symbol,
        feature=feature,
        status=_status(edge_score=edge_score, observations=len(rows)),
        preferred_bucket=preferred_bucket,
        observations=len(rows),
        low_bucket_mean_next_return=low_mean,
        low_bucket_hit_rate=low_hit,
        high_bucket_mean_next_return=high_mean,
        high_bucket_hit_rate=high_hit,
        correlation_to_next_return=correlation,
        edge_score=edge_score,
        next_step=(
            f"rerun {symbol} {feature} on recent windows, then test "
            "funding PnL, fees, spread, and regime split"
        ),
    )


def _preferred_bucket(*, low_mean: float, low_hit: float, high_mean: float, high_hit: float) -> str:
    if high_mean > low_mean and high_hit >= low_hit:
        return "high"
    if low_mean > high_mean and low_hit >= high_hit:
        return "low"
    if high_mean > low_mean:
        return "high_mean_only"
    return "low_mean_only"


def _edge_score(
    *,
    low_mean: float,
    low_hit: float,
    high_mean: float,
    high_hit: float,
    correlation: float,
    observations: int,
) -> float:
    mean_component = abs(high_mean - low_mean) * 10_000.0
    hit_component = abs(high_hit - low_hit) * 25.0
    corr_component = abs(correlation) * 20.0
    sample_component = min(observations / 20.0, 5.0)
    return mean_component + hit_component + corr_component + sample_component


def _status(*, edge_score: float, observations: int) -> str:
    if observations < 40:
        return "thin_symbol_history"
    if edge_score >= 180.0:
        return "symbol_feature_priority"
    if edge_score >= 120.0:
        return "symbol_feature_watch"
    return "weak_symbol_feature_context"


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    denominator = (
        sum((left_value - left_mean) ** 2 for left_value in left)
        * sum((right_value - right_mean) ** 2 for right_value in right)
    ) ** 0.5
    return numerator / denominator if denominator > 0.0 else 0.0


def _hit_rate(values: tuple[float, ...]) -> float:
    return mean(1.0 if value > 0.0 else 0.0 for value in values) if values else 0.0


def _mean(values: tuple[float, ...]) -> float:
    return mean(values) if values else 0.0


def _float(value: object) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-path",
        type=Path,
        default=ROOT / "binance_derivatives_history.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "binance_derivatives_symbol_feature_candidates.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "binance_derivatives_symbol_feature_candidates.md",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    candidates = build_symbol_feature_candidates(history_path=args.history_path)
    write_symbol_feature_candidates_csv(candidates, output_path=args.output_path)
    write_symbol_feature_candidates_md(candidates, output_path=args.markdown_output_path, top=args.top)
    for candidate in candidates[: args.top]:
        print(
            candidate.symbol,
            candidate.feature,
            candidate.status,
            candidate.preferred_bucket,
            f"score={candidate.edge_score:.4f}",
        )


if __name__ == "__main__":
    main()
