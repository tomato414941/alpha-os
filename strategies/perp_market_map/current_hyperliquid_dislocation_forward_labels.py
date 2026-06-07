from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
HOURS_PER_YEAR = 24.0 * 365.0


@dataclass(frozen=True)
class HyperliquidDislocationForwardLabel:
    timestamp: str
    asset: str
    status: str
    side: str
    score: float
    direction: int
    annualized_funding: float
    impact_spread: float
    conservative_cost_bps: float
    raw_return_15m: float | None
    raw_return_1h: float | None
    raw_return_4h: float | None
    directional_return_15m: float | None
    directional_return_1h: float | None
    directional_return_4h: float | None
    funding_return_15m: float
    funding_return_1h: float
    funding_return_4h: float
    net_15m_bps: float | None
    net_1h_bps: float | None
    net_4h_bps: float | None
    outcome_15m: str
    outcome_1h: str
    outcome_4h: str


def build_hyperliquid_dislocation_forward_labels(
    *,
    input_path: Path = ROOT / "current_hyperliquid_dislocation_candidates.csv",
    fee_bps_per_fill: float = 5.0,
) -> tuple[HyperliquidDislocationForwardLabel, ...]:
    rows = _read_rows(input_path)
    candles_by_asset = {
        asset: _fetch_hyperliquid_candles(asset)
        for asset in sorted({row["asset"] for row in rows})
    }
    labels = tuple(
        _build_label(
            row=row,
            candles=candles_by_asset.get(row["asset"], ()),
            fee_bps_per_fill=fee_bps_per_fill,
        )
        for row in rows
    )
    return tuple(sorted(labels, key=_sort_key, reverse=True))


def write_hyperliquid_dislocation_forward_labels_csv(
    labels: tuple[HyperliquidDislocationForwardLabel, ...],
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
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
        )
        for label in labels:
            writer.writerow(
                (
                    label.timestamp,
                    label.asset,
                    label.status,
                    label.side,
                    f"{label.score:.8f}",
                    label.direction,
                    f"{label.annualized_funding:.8f}",
                    f"{label.impact_spread:.12f}",
                    f"{label.conservative_cost_bps:.8f}",
                    _optional_float(label.raw_return_15m),
                    _optional_float(label.raw_return_1h),
                    _optional_float(label.raw_return_4h),
                    _optional_float(label.directional_return_15m),
                    _optional_float(label.directional_return_1h),
                    _optional_float(label.directional_return_4h),
                    f"{label.funding_return_15m:.8f}",
                    f"{label.funding_return_1h:.8f}",
                    f"{label.funding_return_4h:.8f}",
                    _optional_float(label.net_15m_bps),
                    _optional_float(label.net_1h_bps),
                    _optional_float(label.net_4h_bps),
                    label.outcome_15m,
                    label.outcome_1h,
                    label.outcome_4h,
                )
            )
    return output_path


def write_hyperliquid_dislocation_forward_labels_md(
    labels: tuple[HyperliquidDislocationForwardLabel, ...],
    *,
    output_path: Path,
    top: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    covered_15m = tuple(label for label in labels if label.net_15m_bps is not None)
    covered_1h = tuple(label for label in labels if label.net_1h_bps is not None)
    covered_4h = tuple(label for label in labels if label.net_4h_bps is not None)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Current Hyperliquid Dislocation Forward Labels\n\n")
        handle.write(
            "This labels Hyperliquid dislocation candidates after rough taker fees, "
            "impact spread, and funding carry. It is still paper labeling, not a "
            "live fill or deployable strategy.\n\n"
        )
        handle.write(f"- rows: `{len(labels)}`\n")
        handle.write(f"- covered 15m: `{len(covered_15m)}`\n")
        handle.write(f"- covered 1h: `{len(covered_1h)}`\n")
        handle.write(f"- covered 4h: `{len(covered_4h)}`\n\n")
        handle.write(
            "| asset | status | side | score | cost bps | net15 | out15 | net1h | out1h | net4h | out4h |\n"
        )
        handle.write("| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |\n")
        for label in labels[:top]:
            handle.write(
                f"| {label.asset} | "
                f"{label.status} | "
                f"{label.side} | "
                f"{label.score:.4f} | "
                f"{label.conservative_cost_bps:.2f} | "
                f"{_optional_display(label.net_15m_bps)} | "
                f"{label.outcome_15m} | "
                f"{_optional_display(label.net_1h_bps)} | "
                f"{label.outcome_1h} | "
                f"{_optional_display(label.net_4h_bps)} | "
                f"{label.outcome_4h} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Positive net means the candidate side beat rough taker costs plus "
            "the funding carry estimate over that horizon. Pending rows simply "
            "have not had enough elapsed time since the candidate snapshot.\n"
        )
    return output_path


def _build_label(
    *,
    row: dict[str, str],
    candles: tuple[dict[str, float], ...],
    fee_bps_per_fill: float,
) -> HyperliquidDislocationForwardLabel:
    timestamp = _parse_datetime(row["timestamp"])
    direction = _direction_for_side(row["side"])
    raw_return_15m = _forward_return(candles, timestamp, timestamp + timedelta(minutes=15))
    raw_return_1h = _forward_return(candles, timestamp, timestamp + timedelta(hours=1))
    raw_return_4h = _forward_return(candles, timestamp, timestamp + timedelta(hours=4))
    directional_return_15m = _directional_return(raw_return_15m, direction=direction)
    directional_return_1h = _directional_return(raw_return_1h, direction=direction)
    directional_return_4h = _directional_return(raw_return_4h, direction=direction)
    annualized_funding = _float(row.get("annualized_funding"))
    funding_return_15m = _funding_return(annualized_funding, direction=direction, hours=0.25)
    funding_return_1h = _funding_return(annualized_funding, direction=direction, hours=1.0)
    funding_return_4h = _funding_return(annualized_funding, direction=direction, hours=4.0)
    conservative_cost_bps = (fee_bps_per_fill * 2.0) + (_float(row.get("impact_spread")) * 10_000.0)
    net_15m_bps = _net_bps(
        directional_return_15m,
        funding_return=funding_return_15m,
        conservative_cost_bps=conservative_cost_bps,
    )
    net_1h_bps = _net_bps(
        directional_return_1h,
        funding_return=funding_return_1h,
        conservative_cost_bps=conservative_cost_bps,
    )
    net_4h_bps = _net_bps(
        directional_return_4h,
        funding_return=funding_return_4h,
        conservative_cost_bps=conservative_cost_bps,
    )
    return HyperliquidDislocationForwardLabel(
        timestamp=timestamp.isoformat(),
        asset=row["asset"],
        status=row["status"],
        side=row["side"],
        score=_float(row.get("score")),
        direction=direction,
        annualized_funding=annualized_funding,
        impact_spread=_float(row.get("impact_spread")),
        conservative_cost_bps=conservative_cost_bps,
        raw_return_15m=raw_return_15m,
        raw_return_1h=raw_return_1h,
        raw_return_4h=raw_return_4h,
        directional_return_15m=directional_return_15m,
        directional_return_1h=directional_return_1h,
        directional_return_4h=directional_return_4h,
        funding_return_15m=funding_return_15m,
        funding_return_1h=funding_return_1h,
        funding_return_4h=funding_return_4h,
        net_15m_bps=net_15m_bps,
        net_1h_bps=net_1h_bps,
        net_4h_bps=net_4h_bps,
        outcome_15m=_outcome(net_15m_bps, horizon="15m"),
        outcome_1h=_outcome(net_1h_bps, horizon="1h"),
        outcome_4h=_outcome(net_4h_bps, horizon="4h"),
    )


def _fetch_hyperliquid_candles(asset: str) -> tuple[dict[str, float], ...]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=24)
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
            "timestamp": float(candle["t"]),
            "end_timestamp": float(candle["T"]),
            "close": float(candle["c"]),
        }
        for candle in response.json()
    )


def _forward_return(
    candles: tuple[dict[str, float], ...],
    start: datetime,
    target: datetime,
) -> float | None:
    start_close = _close_at_or_after(candles, start)
    end_close = _close_at_or_after(candles, target)
    if start_close is None or end_close is None:
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


def _directional_return(value: float | None, *, direction: int) -> float | None:
    if value is None or direction == 0:
        return None
    return value * direction


def _funding_return(annualized_funding: float, *, direction: int, hours: float) -> float:
    funding_rate_per_hour = annualized_funding / HOURS_PER_YEAR
    return -direction * funding_rate_per_hour * hours


def _net_bps(
    value: float | None,
    *,
    funding_return: float,
    conservative_cost_bps: float,
) -> float | None:
    if value is None:
        return None
    return ((value + funding_return) * 10_000.0) - conservative_cost_bps


def _outcome(value: float | None, *, horizon: str) -> str:
    if value is None:
        return f"pending_{horizon}"
    if value > 0.0:
        return f"paper_{horizon}_win"
    return f"paper_{horizon}_loss"


def _sort_key(label: HyperliquidDislocationForwardLabel) -> tuple[bool, float, bool, float, float]:
    return (
        label.net_15m_bps is not None,
        label.net_15m_bps or -1_000_000.0,
        label.net_1h_bps is not None,
        label.net_1h_bps or -1_000_000.0,
        label.score,
    )


def _direction_for_side(side: str) -> int:
    if side.startswith("long"):
        return 1
    if side.startswith("short"):
        return -1
    return 0


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _float(value: str | None) -> float:
    return float(value) if value else 0.0


def _optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def _optional_display(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_candidates.csv",
    )
    parser.add_argument("--fee-bps-per-fill", type=float, default=5.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_forward_labels.csv",
    )
    parser.add_argument(
        "--md-output-path",
        type=Path,
        default=ROOT / "current_hyperliquid_dislocation_forward_labels.md",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    labels = build_hyperliquid_dislocation_forward_labels(
        input_path=args.input_path,
        fee_bps_per_fill=args.fee_bps_per_fill,
    )
    write_hyperliquid_dislocation_forward_labels_csv(labels, output_path=args.output_path)
    write_hyperliquid_dislocation_forward_labels_md(labels, output_path=args.md_output_path, top=args.top)
    for label in labels[: args.top]:
        print(
            label.asset,
            label.status,
            label.side,
            f"net15={_optional_display(label.net_15m_bps)}",
            label.outcome_15m,
            f"net1h={_optional_display(label.net_1h_bps)}",
            label.outcome_1h,
        )


if __name__ == "__main__":
    main()
