from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parent
STRATEGIES_ROOT = ROOT.parent


@dataclass(frozen=True)
class SignalSample:
    timestamp: datetime
    source: str
    action: str
    asset: str


@dataclass(frozen=True)
class ForwardLabel:
    timestamp: str
    source: str
    action: str
    asset: str
    return_15m: float | None
    return_1h: float | None


@dataclass(frozen=True)
class ForwardLabelSummary:
    source: str
    action: str
    asset: str
    observations: int
    coverage_15m: int
    coverage_1h: int
    mean_return_15m: float | None
    mean_return_1h: float | None
    positive_15m_rate: float | None
    positive_1h_rate: float | None


def build_forward_labels() -> tuple[ForwardLabel, ...]:
    samples = _collect_signal_samples()
    candles_by_asset = {
        asset: _fetch_candles(asset)
        for asset in sorted({sample.asset for sample in samples})
    }
    return tuple(
        _build_forward_label(sample=sample, candles=candles_by_asset.get(sample.asset, ()))
        for sample in samples
    )


def summarize_forward_labels(labels: tuple[ForwardLabel, ...]) -> tuple[ForwardLabelSummary, ...]:
    grouped: dict[tuple[str, str, str], list[ForwardLabel]] = {}
    for label in labels:
        grouped.setdefault((label.source, label.action, label.asset), []).append(label)
    summaries = tuple(
        _summarize_group(key=key, labels=tuple(rows))
        for key, rows in grouped.items()
    )
    return tuple(
        sorted(
            summaries,
            key=lambda row: (
                row.coverage_15m,
                row.mean_return_15m or -1.0,
                row.coverage_1h,
                row.mean_return_1h or -1.0,
            ),
            reverse=True,
        )
    )


def write_forward_labels_csv(labels: tuple[ForwardLabel, ...], *, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("timestamp", "source", "action", "asset", "return_15m", "return_1h"))
        for label in labels:
            writer.writerow(
                (
                    label.timestamp,
                    label.source,
                    label.action,
                    label.asset,
                    "" if label.return_15m is None else f"{label.return_15m:.8f}",
                    "" if label.return_1h is None else f"{label.return_1h:.8f}",
                )
            )
    return output_path


def write_forward_label_summary_csv(
    summaries: tuple[ForwardLabelSummary, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "source",
                "action",
                "asset",
                "observations",
                "coverage_15m",
                "coverage_1h",
                "mean_return_15m",
                "mean_return_1h",
                "positive_15m_rate",
                "positive_1h_rate",
            )
        )
        for row in summaries:
            writer.writerow(
                (
                    row.source,
                    row.action,
                    row.asset,
                    row.observations,
                    row.coverage_15m,
                    row.coverage_1h,
                    "" if row.mean_return_15m is None else f"{row.mean_return_15m:.8f}",
                    "" if row.mean_return_1h is None else f"{row.mean_return_1h:.8f}",
                    "" if row.positive_15m_rate is None else f"{row.positive_15m_rate:.8f}",
                    "" if row.positive_1h_rate is None else f"{row.positive_1h_rate:.8f}",
                )
            )
    return output_path


def write_forward_label_summary_md(
    summaries: tuple[ForwardLabelSummary, ...],
    *,
    output_path: Path,
    top: int = 25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current HL Signal Forward Labels\n\n")
        handle.write(
            "This labels elapsed monitor samples with subsequent Hyperliquid candle "
            "returns. It is a small forward-label check, not a final alpha test.\n\n"
        )
        handle.write(
            "| source | action | asset | obs | cov15 | cov1h | mean 15m | mean 1h | hit15 | hit1h |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries[:top]:
            handle.write(
                "| "
                f"{row.source} | "
                f"{row.action} | "
                f"{row.asset} | "
                f"{row.observations} | "
                f"{row.coverage_15m} | "
                f"{row.coverage_1h} | "
                f"{'' if row.mean_return_15m is None else f'{row.mean_return_15m:.6f}'} | "
                f"{'' if row.mean_return_1h is None else f'{row.mean_return_1h:.6f}'} | "
                f"{'' if row.positive_15m_rate is None else f'{row.positive_15m_rate:.6f}'} | "
                f"{'' if row.positive_1h_rate is None else f'{row.positive_1h_rate:.6f}'} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This labels price movement after signal timestamps only. It does not yet "
            "include funding PnL, hedge PnL, fees, adverse selection, or neutral baselines.\n"
        )
    return output_path


def _collect_signal_samples() -> tuple[SignalSample, ...]:
    return _perp_crowding_samples() + _cross_exchange_samples()


def _perp_crowding_samples() -> tuple[SignalSample, ...]:
    path = STRATEGIES_ROOT / "perp_market_map" / "current_crowding_reversion_monitor_samples.csv"
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(
            SignalSample(
                timestamp=_parse_datetime(row["timestamp"]),
                source="perp_carry_reversion",
                action=row["action"],
                asset=row["asset"],
            )
            for row in csv.DictReader(handle)
        )


def _cross_exchange_samples() -> tuple[SignalSample, ...]:
    path = STRATEGIES_ROOT / "cross_exchange_funding" / "stable_12_sample_monitor_samples.csv"
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(
            SignalSample(
                timestamp=_parse_datetime(row["timestamp"]),
                source=row["source"],
                action=row["action"],
                asset=row["asset"],
            )
            for row in csv.DictReader(handle)
            if row["source"] == "okx_hl_current"
        )


def _build_forward_label(
    *,
    sample: SignalSample,
    candles: tuple[dict[str, float], ...],
) -> ForwardLabel:
    start_close = _close_at_or_after(candles, sample.timestamp)
    return ForwardLabel(
        timestamp=sample.timestamp.isoformat(),
        source=sample.source,
        action=sample.action,
        asset=sample.asset,
        return_15m=_forward_return(
            candles=candles,
            start_close=start_close,
            target=sample.timestamp + timedelta(minutes=15),
        ),
        return_1h=_forward_return(
            candles=candles,
            start_close=start_close,
            target=sample.timestamp + timedelta(hours=1),
        ),
    )


def _forward_return(
    *,
    candles: tuple[dict[str, float], ...],
    start_close: float | None,
    target: datetime,
) -> float | None:
    if start_close is None:
        return None
    end_close = _close_at_or_after(candles, target)
    if end_close is None:
        return None
    return (end_close / start_close) - 1.0 if start_close > 0.0 else None


def _close_at_or_after(candles: tuple[dict[str, float], ...], target: datetime) -> float | None:
    target_ms = target.timestamp() * 1000
    for candle in candles:
        if candle["timestamp"] <= target_ms <= candle["end_timestamp"]:
            return candle["close"]
        if candle["timestamp"] >= target_ms:
            return candle["close"]
    return None


def _fetch_candles(asset: str) -> tuple[dict[str, float], ...]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=8)
    response = requests.post(
        HYPERLIQUID_INFO_URL,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": asset,
                "interval": "15m",
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    return tuple(
        {
            "timestamp": float(row["t"]),
            "end_timestamp": float(row["T"]),
            "close": float(row["c"]),
        }
        for row in response.json()
    )


def _summarize_group(
    *,
    key: tuple[str, str, str],
    labels: tuple[ForwardLabel, ...],
) -> ForwardLabelSummary:
    returns_15m = tuple(row.return_15m for row in labels if row.return_15m is not None)
    returns_1h = tuple(row.return_1h for row in labels if row.return_1h is not None)
    return ForwardLabelSummary(
        source=key[0],
        action=key[1],
        asset=key[2],
        observations=len(labels),
        coverage_15m=len(returns_15m),
        coverage_1h=len(returns_1h),
        mean_return_15m=_mean_or_none(returns_15m),
        mean_return_1h=_mean_or_none(returns_1h),
        positive_15m_rate=_positive_rate_or_none(returns_15m),
        positive_1h_rate=_positive_rate_or_none(returns_1h),
    )


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def _positive_rate_or_none(values: tuple[float, ...]) -> float | None:
    return sum(value > 0.0 for value in values) / len(values) if values else None


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-output-path",
        type=Path,
        default=ROOT / "current_hl_signal_forward_labels.csv",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=ROOT / "current_hl_signal_forward_label_summary.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hl_signal_forward_label_summary.md",
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    labels = build_forward_labels()
    summaries = summarize_forward_labels(labels)
    write_forward_labels_csv(labels, output_path=args.labels_output_path)
    write_forward_label_summary_csv(summaries, output_path=args.summary_output_path)
    write_forward_label_summary_md(summaries, output_path=args.md_output_path, top=args.top)
    for row in summaries[: args.top]:
        print(
            row.source,
            row.action,
            row.asset,
            f"cov15={row.coverage_15m}",
            f"mean15={'' if row.mean_return_15m is None else f'{row.mean_return_15m:.4f}'}",
            f"cov1h={row.coverage_1h}",
        )


if __name__ == "__main__":
    main()
