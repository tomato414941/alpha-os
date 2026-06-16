from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-tensor", type=Path, required=True)
    parser.add_argument("--sparse-store", type=Path, required=True)
    parser.add_argument("--lookback-hours", type=float)
    parser.add_argument("--max-events-per-sample", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def parse_interval_hours(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([mhd])", value.strip())
    if match is None:
        raise ValueError(f"unsupported interval: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return amount / 60.0
    if unit == "h":
        return amount
    if unit == "d":
        return amount * 24.0
    raise AssertionError(unit)


def sample_times_ns(price_tensor: np.lib.npyio.NpzFile) -> np.ndarray:
    timestamps = pd.to_datetime(price_tensor["sample_times"], utc=True, format="mixed")
    return (
        timestamps.tz_convert("UTC")
        .tz_localize(None)
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy()
    )


def inferred_lookback_hours(price_tensor: np.lib.npyio.NpzFile) -> float:
    lookback_steps = int(price_tensor["lookback"][0])
    interval = str(price_tensor["interval"][0])
    return lookback_steps * parse_interval_hours(interval)


def percentile_lines(name: str, values: np.ndarray) -> list[str]:
    percentiles = np.percentile(values, [0, 25, 50, 75, 95, 99, 100])
    labels = ["min", "p25", "median", "p75", "p95", "p99", "max"]
    return [f"- {name}_{label}: {value:.2f}" for label, value in zip(labels, percentiles)]


def top_id_lines(
    title: str,
    ids: np.ndarray,
    names: np.ndarray,
    limit: int = 20,
) -> list[str]:
    counts = np.bincount(ids, minlength=len(names))
    top = np.argsort(counts)[::-1][:limit]
    lines = [f"## {title}", ""]
    for rank, idx in enumerate(top, start=1):
        count = int(counts[idx])
        if count == 0:
            continue
        lines.append(f"{rank}. `{names[idx]}`: {count}")
    return lines


def main() -> None:
    args = parse_args()
    if args.max_events_per_sample is not None and args.max_events_per_sample <= 0:
        raise ValueError("--max-events-per-sample must be positive")

    price_tensor = np.load(args.price_tensor, allow_pickle=False)
    sparse_store = np.load(args.sparse_store, allow_pickle=False)

    sample_ns = sample_times_ns(price_tensor)
    event_time_ns = sparse_store["event_time_ns"]
    stream_id = sparse_store["stream_id"]
    field_id = sparse_store["field_id"]
    stream_names = sparse_store["stream_names"]
    field_names = sparse_store["field_names"]

    lookback_hours = args.lookback_hours
    if lookback_hours is None:
        lookback_hours = inferred_lookback_hours(price_tensor)
    lookback_ns = int(dt.timedelta(hours=lookback_hours).total_seconds() * 1_000_000_000)

    window_start_time_ns = sample_ns - lookback_ns
    window_start = np.searchsorted(event_time_ns, window_start_time_ns, side="left").astype(
        np.int64
    )
    window_end = np.searchsorted(event_time_ns, sample_ns, side="right").astype(np.int64)
    counts = window_end - window_start

    truncated_events = np.zeros_like(counts)
    if args.max_events_per_sample is not None:
        truncated_events = np.maximum(counts - args.max_events_per_sample, 0)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            sample_times=price_tensor["sample_times"],
            window_start=window_start,
            window_end=window_end,
            lookback_ns=np.array([lookback_ns], dtype=np.int64),
            sparse_store=np.array([str(args.sparse_store)]),
            created_at=np.array([dt.datetime.now(dt.UTC).isoformat()]),
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# Sparse Event Window Profile",
        "",
        f"- price_tensor: {args.price_tensor}",
        f"- sparse_store: {args.sparse_store}",
        f"- output: {args.output}" if args.output is not None else "- output: none",
        f"- samples: {len(sample_ns)}",
        f"- lookback_hours: {lookback_hours:.2f}",
        f"- sparse_values: {len(event_time_ns)}",
        f"- zero_event_samples: {int((counts == 0).sum())}",
        f"- events_per_sample_mean: {float(counts.mean()):.2f}",
        *percentile_lines("events_per_sample", counts),
    ]
    if args.max_events_per_sample is not None:
        retained = np.minimum(counts, args.max_events_per_sample).sum()
        total = counts.sum()
        retained_fraction = float(retained / total) if total else 1.0
        summary_lines.extend(
            [
                f"- max_events_per_sample: {args.max_events_per_sample}",
                f"- samples_above_max_events: {int((truncated_events > 0).sum())}",
                f"- truncated_events: {int(truncated_events.sum())}",
                f"- retained_event_fraction_if_truncated: {retained_fraction:.4f}",
            ]
        )
    summary_lines.extend(
        [
            "",
            "## Guard",
            "",
            "Window arrays are slice indexes into the sparse store.",
            "They do not materialize signal streams as dense tensor features.",
            "",
            *top_id_lines("Top Streams", stream_id, stream_names),
            "",
            *top_id_lines("Top Fields", field_id, field_names),
            "",
        ]
    )
    args.summary.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"wrote {args.summary}")
    if args.output is not None:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
