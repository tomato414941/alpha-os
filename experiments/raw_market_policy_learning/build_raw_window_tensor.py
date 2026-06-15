from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


API = "https://data-api.binance.vision/api/v3"
RAW_KLINE_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
]
SIGNAL_NOISE_ID_COLUMNS = {"signal_name", "timestamp", "date", "name"}


def fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "alpha-os-raw-window/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read())


def fetch_exchange_symbols(quote_asset: str) -> list[str]:
    payload = fetch_json(f"{API}/exchangeInfo", timeout=30)
    symbols = []
    for row in payload.get("symbols", []):
        if row.get("status") != "TRADING":
            continue
        if row.get("quoteAsset") != quote_asset:
            continue
        if not bool(row.get("isSpotTradingAllowed", False)):
            continue
        symbol = str(row.get("symbol", "")).strip()
        if symbol:
            symbols.append(symbol)
    return sorted(set(symbols))


def requested_symbols(symbols: list[str], sample_symbols: int | None, sample_seed: int) -> list[str]:
    if sample_symbols is None or sample_symbols >= len(symbols):
        return symbols
    rng = random.Random(sample_seed)
    return sorted(rng.sample(symbols, sample_symbols))


def fetch_klines(symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    rows: list[list[object]] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while start_ms < end_ms:
        qs = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "limit": 1000,
                "startTime": start_ms,
                "endTime": end_ms,
            }
        )
        batch = fetch_json(f"{API}/klines?{qs}")
        if not batch:
            break
        rows.extend(batch)
        next_ms = int(batch[-1][0]) + 1
        if next_ms <= start_ms:
            break
        start_ms = next_ms
        time.sleep(0.03)
    if len(rows) < 200:
        raise RuntimeError(f"insufficient rows for {symbol}: {len(rows)}")
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        df[col] = df[col].astype(float)
    df["trade_count"] = df["trade_count"].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column in RAW_KLINE_FEATURES:
        out[column] = df[column]
    return out


def feature_name(*parts: str) -> str:
    raw = "_".join(part for part in parts if part)
    return re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_").lower()


def load_signal_noise_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "signal_name" not in frame.columns:
        raise ValueError("signal-noise streams CSV must contain signal_name")
    timestamp_column = "timestamp" if "timestamp" in frame.columns else "date"
    if timestamp_column not in frame.columns:
        raise ValueError("signal-noise streams CSV must contain timestamp or date")

    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, format="mixed")
    numeric_columns = []
    for column in frame.columns:
        if column in SIGNAL_NOISE_ID_COLUMNS:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            frame[column] = values
            numeric_columns.append(column)
    if not numeric_columns:
        raise ValueError(f"no numeric signal-noise stream columns found in {path}")

    features: dict[str, pd.Series] = {}
    for signal_name, group in frame.groupby("signal_name", sort=True):
        group = group.sort_values(timestamp_column).set_index(timestamp_column)
        group = group[~group.index.duplicated(keep="last")]
        for column in numeric_columns:
            series = group[column].dropna()
            if series.empty:
                continue
            features[feature_name("signal_noise", str(signal_name), column)] = series
    if not features:
        raise ValueError(f"no usable signal-noise stream values found in {path}")
    return pd.DataFrame(features).sort_index()


def build_windows(
    feature_by_symbol: dict[str, pd.DataFrame],
    lookback: int,
    horizons: list[int],
    cost_bps: float,
    signal_noise_features: pd.DataFrame | None = None,
):
    instruments = sorted(feature_by_symbol)
    common_index = None
    for df in feature_by_symbol.values():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None:
        raise RuntimeError("no index")
    common_index = common_index.sort_values()

    aligned_signal_noise = None
    signal_noise_feature_names: list[str] = []
    if signal_noise_features is not None:
        aligned_signal_noise = signal_noise_features.ffill().reindex(common_index, method="ffill")
        signal_noise_feature_names = list(aligned_signal_noise.columns)

    panels = []
    closes = []
    for instrument in instruments:
        df = feature_by_symbol[instrument].reindex(common_index)
        instrument_features = df[RAW_KLINE_FEATURES].to_numpy(np.float32)
        if aligned_signal_noise is not None:
            instrument_features = np.concatenate(
                [
                    instrument_features,
                    aligned_signal_noise.to_numpy(np.float32),
                ],
                axis=1,
            )
        panels.append(instrument_features)
        closes.append(df["close"].to_numpy(np.float32))
    values = np.stack(panels, axis=0)  # instruments, time, raw_features
    close_values = np.stack(closes, axis=0)  # instruments, time

    max_horizon = max(horizons)
    valid_times = []
    windows = []
    rewards = []
    cost = cost_bps / 10_000.0

    for t in range(lookback, len(common_index) - max_horizon):
        window = values[:, t - lookback : t, :]
        if not np.isfinite(window).all():
            continue
        reward_parts = []
        for horizon in horizons:
            future_return = np.log(close_values[:, t + horizon] / close_values[:, t])
            long_reward = future_return - cost
            short_reward = -future_return - cost
            flat_reward = np.zeros_like(future_return)
            reward_parts.append(np.stack([long_reward, short_reward, flat_reward], axis=1))
        reward = np.stack(reward_parts, axis=1)  # instruments, horizons, actions
        if not np.isfinite(reward).all():
            continue
        valid_times.append(common_index[t])
        windows.append(window)
        rewards.append(reward.astype(np.float32))

    if not windows:
        raise RuntimeError("no valid windows after aligning features and rewards")
    feature_names = RAW_KLINE_FEATURES + signal_noise_feature_names
    return instruments, common_index, np.stack(windows), np.stack(rewards), valid_times, feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quote-asset", default="USDT")
    parser.add_argument("--sample-symbols", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--lookback", type=int, default=72)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizons", default="1,4,24")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--signal-noise-streams", type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    created_at = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
    horizons = [int(x) for x in args.horizons.split(",")]
    end = created_at - dt.timedelta(hours=max(horizons))
    start = end - dt.timedelta(days=args.days + 5)
    symbol_inventory = fetch_exchange_symbols(args.quote_asset)
    symbols = requested_symbols(symbol_inventory, args.sample_symbols, args.sample_seed)
    print(
        f"loaded exchange symbols quote_asset={args.quote_asset} "
        f"inventory={len(symbol_inventory)} requested={len(symbols)}",
        flush=True,
    )

    feature_by_symbol = {}
    failures = []
    for i, symbol in enumerate(symbols, 1):
        try:
            df = fetch_klines(symbol, args.interval, start, end)
            feature_by_symbol[symbol.removesuffix("USDT")] = raw_features(df)
            print(f"{i:03d}/{len(symbols)} {symbol} rows={len(df)}", flush=True)
        except Exception as exc:
            failures.append((symbol, str(exc)))
            print(f"{i:03d}/{len(symbols)} {symbol} failed: {exc}", flush=True)

    signal_noise_features = None
    if args.signal_noise_streams is not None:
        signal_noise_features = load_signal_noise_features(args.signal_noise_streams)
        print(
            f"loaded signal-noise features={len(signal_noise_features.columns)} "
            f"rows={len(signal_noise_features)}",
            flush=True,
        )

    instruments, common_index, x, reward, sample_times, feature_names = build_windows(
        feature_by_symbol,
        args.lookback,
        horizons,
        args.cost_bps,
        signal_noise_features,
    )
    action_names = np.array(["long", "short", "flat"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        reward=reward,
        sample_times=np.array([str(t) for t in sample_times]),
        instruments=np.array(instruments),
        raw_features=np.array(feature_names),
        horizons=np.array(horizons),
        actions=action_names,
        created_at=np.array([created_at.isoformat()]),
        interval=np.array([args.interval]),
        lookback=np.array([args.lookback]),
        cost_bps=np.array([args.cost_bps]),
    )

    flat = reward.reshape(-1, len(horizons), len(action_names))
    means = flat.mean(axis=0)
    lines = []
    for hi, horizon in enumerate(horizons):
        for ai, action in enumerate(action_names):
            lines.append(f"{horizon},{action},{means[hi, ai]:.8f}")

    summary = Path(args.summary)
    summary.write_text(
        "\n".join(
            [
                "# Raw Window Tensor Dataset",
                "",
                f"- created_at: {created_at.isoformat()}",
                f"- interval: {args.interval}",
                f"- quote_asset: {args.quote_asset}",
                f"- symbol_inventory: {len(symbol_inventory)}",
                f"- requested_symbols: {len(symbols)}",
                f"- sample_symbols: {args.sample_symbols or 'none'}",
                f"- sample_seed: {args.sample_seed}",
                f"- instruments: {len(instruments)}",
                f"- days: {args.days}",
                f"- lookback_steps: {args.lookback}",
                f"- raw_features: {len(feature_names)}",
                f"- signal_noise_streams: {args.signal_noise_streams or 'none'}",
                f"- signal_noise_features: {0 if signal_noise_features is None else len(signal_noise_features.columns)}",
                f"- horizons_hours: {', '.join(map(str, horizons))}",
                f"- cost_bps: {args.cost_bps}",
                f"- samples: {x.shape[0]}",
                f"- x_shape: {x.shape}",
                f"- reward_shape: {reward.shape}",
                f"- output: {output}",
                "",
                "## Instruments",
                "",
                ", ".join(instruments),
                "",
                "## Features",
                "",
                ", ".join(feature_names),
                "",
                "## Mean Reward",
                "",
                "```csv",
                "horizon,action,mean_reward",
                "\n".join(lines),
                "```",
                "",
                "## Fetch Failures",
                "",
                "```text",
                "\n".join(f"{s}: {e}" for s, e in failures) if failures else "none",
                "```",
                "",
                "## Guard",
                "",
                "This dataset keeps raw-ish rolling market windows for sequence models.",
                "It is not a strategy and does not define alpha rules.",
                "Signal-noise streams are aligned by timestamp with forward fill; this",
                "experiment does not yet have signal-level available_at metadata.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
