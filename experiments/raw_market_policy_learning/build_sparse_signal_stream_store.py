from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = {"signal_name", "timestamp", "date", "name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--price-tensor", type=Path)
    parser.add_argument("--keep-after-price-end", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def id_for(mapping: dict[str, int], value: str) -> int:
    found = mapping.get(value)
    if found is not None:
        return found
    found = len(mapping)
    mapping[value] = found
    return found


def append_numeric_column(
    *,
    frame: pd.DataFrame,
    timestamps_ns: np.ndarray,
    column: str,
    field_to_id: dict[str, int],
    stream_to_id: dict[str, int],
    event_time_parts: list[np.ndarray],
    stream_id_parts: list[np.ndarray],
    field_id_parts: list[np.ndarray],
    value_parts: list[np.ndarray],
) -> int:
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = values.notna().to_numpy()
    if not mask.any():
        return 0

    field_id = id_for(field_to_id, column)
    names = frame.loc[mask, "signal_name"].astype(str)
    stream_ids = np.array([id_for(stream_to_id, name) for name in names], dtype=np.int32)
    count = int(mask.sum())

    event_time_parts.append(timestamps_ns[mask])
    stream_id_parts.append(stream_ids)
    field_id_parts.append(np.full(count, field_id, dtype=np.int32))
    value_parts.append(values.loc[mask].astype(np.float32).to_numpy())
    return count


def load_price_tensor_info(path: Path | None) -> tuple[dict[str, object], pd.Timestamp | None]:
    if path is None:
        return {}, None
    data = np.load(path)
    info: dict[str, object] = {
        "price_tensor": str(path),
        "price_x_shape": tuple(int(v) for v in data["x"].shape),
        "price_reward_shape": tuple(int(v) for v in data["reward"].shape),
    }
    price_end = None
    if "sample_times" in data:
        sample_times = pd.to_datetime(data["sample_times"], utc=True, format="mixed")
        price_end = sample_times.max()
        info["sample_count"] = int(len(data["sample_times"]))
        info["sample_time_min"] = str(sample_times.min())
        info["sample_time_max"] = str(price_end)
    return info, price_end


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    price_info, price_end = load_price_tensor_info(args.price_tensor)
    clip_after_price_end = price_end is not None and not args.keep_after_price_end
    stream_to_id: dict[str, int] = {}
    field_to_id: dict[str, int] = {}
    event_time_parts: list[np.ndarray] = []
    stream_id_parts: list[np.ndarray] = []
    field_id_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    source_rows = 0
    usable_rows = 0
    clipped_future_rows = 0
    sparse_values = 0

    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None

    for chunk in pd.read_csv(args.streams, chunksize=args.chunk_size, low_memory=False):
        source_rows += len(chunk)
        timestamp_column = "timestamp" if "timestamp" in chunk.columns else "date"
        if "signal_name" not in chunk.columns or timestamp_column not in chunk.columns:
            raise ValueError("streams CSV must contain signal_name and timestamp/date")

        timestamps = pd.to_datetime(
            chunk[timestamp_column],
            utc=True,
            format="mixed",
            errors="coerce",
        )
        valid = timestamps.notna() & chunk["signal_name"].notna()
        if clip_after_price_end:
            future = timestamps > price_end
            clipped_future_rows += int((valid & future).sum())
            valid &= ~future
        if not valid.any():
            continue

        frame = chunk.loc[valid].copy()
        timestamps = timestamps.loc[valid]
        timestamps_ns = (
            timestamps.dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
            .astype("int64")
            .to_numpy()
        )
        usable_rows += len(frame)
        current_min = timestamps.min()
        current_max = timestamps.max()
        timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
        timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)

        for column in frame.columns:
            if column in ID_COLUMNS:
                continue
            sparse_values += append_numeric_column(
                frame=frame,
                timestamps_ns=timestamps_ns,
                column=column,
                field_to_id=field_to_id,
                stream_to_id=stream_to_id,
                event_time_parts=event_time_parts,
                stream_id_parts=stream_id_parts,
                field_id_parts=field_id_parts,
                value_parts=value_parts,
            )

    if sparse_values == 0:
        raise RuntimeError("no numeric signal stream values found")

    event_time_ns = np.concatenate(event_time_parts).astype(np.int64)
    stream_id = np.concatenate(stream_id_parts).astype(np.int32)
    field_id = np.concatenate(field_id_parts).astype(np.int32)
    value = np.concatenate(value_parts).astype(np.float32)
    order = np.lexsort((field_id, stream_id, event_time_ns))

    stream_names = np.array(
        [name for name, _ in sorted(stream_to_id.items(), key=lambda item: item[1])],
    )
    field_names = np.array(
        [name for name, _ in sorted(field_to_id.items(), key=lambda item: item[1])],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_time_ns=event_time_ns[order],
        stream_id=stream_id[order],
        field_id=field_id[order],
        value=value[order],
        stream_names=stream_names,
        field_names=field_names,
        source_streams=np.array([str(args.streams)]),
        created_at=np.array([dt.datetime.now(dt.UTC).isoformat()]),
    )

    dense_estimate_lines: list[str] = []
    if "price_x_shape" in price_info:
        samples, instruments, lookback, price_features = price_info["price_x_shape"]  # type: ignore[misc]
        total_features = int(price_features) + len(stream_to_id) * len(field_to_id)
        dense_gb = samples * instruments * lookback * total_features * 4 / (1024 ** 3)
        dense_estimate_lines = [
            f"- dense_total_features_if_expanded: {total_features}",
            f"- dense_x_float32_gb_if_expanded: {dense_gb:.2f}",
        ]

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "# Sparse Signal Stream Store",
                "",
                f"- source_streams: {args.streams}",
                f"- source_rows: {source_rows}",
                f"- usable_rows: {usable_rows}",
                f"- clipped_future_rows: {clipped_future_rows}",
                f"- sparse_values: {sparse_values}",
                f"- streams: {len(stream_to_id)}",
                f"- fields: {len(field_to_id)}",
                f"- timestamp_min: {timestamp_min}",
                f"- timestamp_max: {timestamp_max}",
                f"- output: {args.output}",
                f"- clip_after_price_end: {clip_after_price_end}",
                *[
                    f"- {key}: {value}"
                    for key, value in price_info.items()
                    if key != "price_tensor"
                ],
                *dense_estimate_lines,
                "",
                "## Fields",
                "",
                ", ".join(field_names.astype(str)),
                "",
                "## Guard",
                "",
                "This store keeps signal-noise observations as timestamped sparse values.",
                "It does not expand global streams across instruments, samples, or lookback windows.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
