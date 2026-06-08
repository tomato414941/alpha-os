from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FeatureRegimeComparison:
    symbol: str
    feature: str
    status: str
    historical_bucket: str
    recent_bucket: str
    historical_score: float
    recent_score: float
    combined_score: float
    historical_mean_edge: float
    recent_mean_edge: float
    historical_hit_edge: float
    recent_hit_edge: float
    next_step: str


def build_feature_regime_comparisons(
    *,
    historical_path: Path,
    recent_path: Path,
) -> tuple[FeatureRegimeComparison, ...]:
    historical_rows = {
        (row.get("symbol", ""), row.get("feature", "")): row
        for row in _read_rows(historical_path)
        if row.get("symbol") and row.get("feature")
    }
    recent_rows = {
        (row.get("symbol", ""), row.get("feature", "")): row
        for row in _read_rows(recent_path)
        if row.get("symbol") and row.get("feature")
    }
    keys = sorted(set(historical_rows) | set(recent_rows))
    comparisons = [
        _comparison_for_key(
            symbol=symbol,
            feature=feature,
            historical=historical_rows.get((symbol, feature), {}),
            recent=recent_rows.get((symbol, feature), {}),
        )
        for symbol, feature in keys
    ]
    return tuple(sorted(comparisons, key=lambda row: row.combined_score, reverse=True))


def write_feature_regime_comparisons_csv(
    rows: tuple[FeatureRegimeComparison, ...],
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
                "historical_bucket",
                "recent_bucket",
                "historical_score",
                "recent_score",
                "combined_score",
                "historical_mean_edge",
                "recent_mean_edge",
                "historical_hit_edge",
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
                    row.historical_bucket,
                    row.recent_bucket,
                    f"{row.historical_score:.8f}",
                    f"{row.recent_score:.8f}",
                    f"{row.combined_score:.8f}",
                    f"{row.historical_mean_edge:.12f}",
                    f"{row.recent_mean_edge:.12f}",
                    f"{row.historical_hit_edge:.8f}",
                    f"{row.recent_hit_edge:.8f}",
                    row.next_step,
                )
            )
    return output_path


def write_feature_regime_comparisons_md(
    rows: tuple[FeatureRegimeComparison, ...],
    *,
    output_path: Path,
    top: int = 60,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Binance Derivatives Feature Regime Compare\n\n")
        handle.write(
            "This compares the 2024Q1 Binance USD-M symbol-feature queue with the recent-window queue. "
            "It separates candidates that persist across regimes from candidates that appear only in the recent panel.\n\n"
        )
        handle.write(
            "| symbol | feature | status | historical bucket | recent bucket | historical score | recent score | combined score | next step |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows[:top]:
            handle.write(
                f"| {row.symbol} | {row.feature} | {row.status} | "
                f"{row.historical_bucket} | {row.recent_bucket} | "
                f"{row.historical_score:.4f} | {row.recent_score:.4f} | "
                f"{row.combined_score:.4f} | {row.next_step} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "`persistent_symbol_feature` rows deserve the next recent-data rerun and execution check. "
            "`recent_symbol_feature_priority` rows are newer regime candidates. "
            "`bucket_regime_shift` rows may still matter, but the direction changed and should not be promoted blindly.\n"
        )
    return output_path


def _comparison_for_key(
    *,
    symbol: str,
    feature: str,
    historical: dict[str, str],
    recent: dict[str, str],
) -> FeatureRegimeComparison:
    historical_score = _float(historical.get("edge_score"))
    recent_score = _float(recent.get("edge_score"))
    historical_bucket = historical.get("preferred_bucket", "")
    recent_bucket = recent.get("preferred_bucket", "")
    historical_mean_edge = _mean_edge(historical)
    recent_mean_edge = _mean_edge(recent)
    historical_hit_edge = _hit_edge(historical)
    recent_hit_edge = _hit_edge(recent)
    status = _status(
        historical_score=historical_score,
        recent_score=recent_score,
        historical_bucket=historical_bucket,
        recent_bucket=recent_bucket,
    )
    combined_score = _combined_score(
        status=status,
        historical_score=historical_score,
        recent_score=recent_score,
        historical_mean_edge=historical_mean_edge,
        recent_mean_edge=recent_mean_edge,
    )
    return FeatureRegimeComparison(
        symbol=symbol,
        feature=feature,
        status=status,
        historical_bucket=historical_bucket or "-",
        recent_bucket=recent_bucket or "-",
        historical_score=historical_score,
        recent_score=recent_score,
        combined_score=combined_score,
        historical_mean_edge=historical_mean_edge,
        recent_mean_edge=recent_mean_edge,
        historical_hit_edge=historical_hit_edge,
        recent_hit_edge=recent_hit_edge,
        next_step=_next_step(symbol=symbol, feature=feature, status=status),
    )


def _status(
    *,
    historical_score: float,
    recent_score: float,
    historical_bucket: str,
    recent_bucket: str,
) -> str:
    if historical_score >= 120.0 and recent_score >= 120.0:
        if _bucket_family(historical_bucket) == _bucket_family(recent_bucket):
            return "persistent_symbol_feature"
        return "bucket_regime_shift"
    if recent_score >= 180.0:
        return "recent_symbol_feature_priority"
    if recent_score >= 120.0:
        return "recent_symbol_feature_watch"
    if historical_score >= 180.0:
        return "historical_symbol_feature_only"
    return "weak_or_unconfirmed"


def _combined_score(
    *,
    status: str,
    historical_score: float,
    recent_score: float,
    historical_mean_edge: float,
    recent_mean_edge: float,
) -> float:
    persistence_bonus = {
        "persistent_symbol_feature": 80.0,
        "bucket_regime_shift": 25.0,
        "recent_symbol_feature_priority": 45.0,
        "recent_symbol_feature_watch": 20.0,
        "historical_symbol_feature_only": -20.0,
    }.get(status, -60.0)
    sign_bonus = 20.0 if historical_mean_edge * recent_mean_edge > 0.0 else 0.0
    return max(historical_score, 0.0) * 0.35 + max(recent_score, 0.0) * 0.65 + persistence_bonus + sign_bonus


def _next_step(*, symbol: str, feature: str, status: str) -> str:
    if status == "persistent_symbol_feature":
        return f"rerun {symbol} {feature} with recent intraday labels, then add execution and funding PnL"
    if status == "bucket_regime_shift":
        return f"split {symbol} {feature} by market regime before using the feature direction"
    if status.startswith("recent_symbol_feature"):
        return f"extend recent {symbol} {feature} window and check whether the effect survives costs"
    if status == "historical_symbol_feature_only":
        return f"deprioritize {symbol} {feature} until it reappears in recent data"
    return f"keep {symbol} {feature} as context only"


def _bucket_family(value: str) -> str:
    if value.startswith("high"):
        return "high"
    if value.startswith("low"):
        return "low"
    return value


def _mean_edge(row: dict[str, str]) -> float:
    return _float(row.get("high_bucket_mean_next_return")) - _float(row.get("low_bucket_mean_next_return"))


def _hit_edge(row: dict[str, str]) -> float:
    return _float(row.get("high_bucket_hit_rate")) - _float(row.get("low_bucket_hit_rate"))


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
        "--historical-path",
        type=Path,
        default=ROOT / "binance_derivatives_symbol_feature_candidates.csv",
    )
    parser.add_argument(
        "--recent-path",
        type=Path,
        default=ROOT / "binance_derivatives_recent_symbol_feature_candidates.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "binance_derivatives_feature_regime_compare.csv",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=ROOT / "binance_derivatives_feature_regime_compare.md",
    )
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    rows = build_feature_regime_comparisons(
        historical_path=args.historical_path,
        recent_path=args.recent_path,
    )
    write_feature_regime_comparisons_csv(rows, output_path=args.output_path)
    write_feature_regime_comparisons_md(rows, output_path=args.markdown_output_path, top=args.top)
    for row in rows[: args.top]:
        print(
            row.symbol,
            row.feature,
            row.status,
            f"historical={row.historical_score:.4f}",
            f"recent={row.recent_score:.4f}",
            f"combined={row.combined_score:.4f}",
        )


if __name__ == "__main__":
    main()
