from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


API = "https://data-api.binance.vision/api/v3"
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT", "SUIUSDT", "LTCUSDT",
    "BCHUSDT", "DOTUSDT", "UNIUSDT", "AAVEUSDT", "NEARUSDT", "APTUSDT",
    "ICPUSDT", "ETCUSDT", "FILUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "ATOMUSDT", "TIAUSDT", "SEIUSDT", "WIFUSDT", "PEPEUSDT", "FETUSDT",
    "RENDERUSDT", "TONUSDT", "HBARUSDT", "XLMUSDT", "ALGOUSDT",
    "VETUSDT", "GRTUSDT", "ENAUSDT", "WLDUSDT",
]
RAW_FEATURES = [
    "log_return",
    "high_rel_close",
    "low_rel_close",
    "log_quote_volume",
    "log_trade_count",
    "taker_buy_quote_share",
]


def fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "alpha-os-raw-window/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read())


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
    for col in ["open", "high", "low", "close", "quote_volume", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["trade_count"] = df["trade_count"].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["log_return"] = np.log(close / close.shift(1))
    out["high_rel_close"] = np.log(df["high"] / close)
    out["low_rel_close"] = np.log(df["low"] / close)
    out["log_quote_volume"] = np.log1p(df["quote_volume"])
    out["log_trade_count"] = np.log1p(df["trade_count"])
    out["taker_buy_quote_share"] = df["taker_buy_quote"] / df["quote_volume"].replace(0, np.nan)
    return out


def build_windows(
    feature_by_symbol: dict[str, pd.DataFrame],
    lookback: int,
    horizons: list[int],
    cost_bps: float,
):
    instruments = sorted(feature_by_symbol)
    common_index = None
    for df in feature_by_symbol.values():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None:
        raise RuntimeError("no index")
    common_index = common_index.sort_values()

    panels = []
    closes = []
    for instrument in instruments:
        df = feature_by_symbol[instrument].reindex(common_index)
        panels.append(df[RAW_FEATURES].to_numpy(np.float32))
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

    return instruments, common_index, np.stack(windows), np.stack(rewards), valid_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=30)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--lookback", type=int, default=72)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizons", default="1,4,24")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    created_at = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
    horizons = [int(x) for x in args.horizons.split(",")]
    end = created_at - dt.timedelta(hours=max(horizons))
    start = end - dt.timedelta(days=args.days + 5)
    symbols = DEFAULT_SYMBOLS[: args.symbols]

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

    instruments, common_index, x, reward, sample_times = build_windows(
        feature_by_symbol,
        args.lookback,
        horizons,
        args.cost_bps,
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
        raw_features=np.array(RAW_FEATURES),
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
                f"- instruments: {len(instruments)}",
                f"- days: {args.days}",
                f"- lookback_steps: {args.lookback}",
                f"- raw_features: {len(RAW_FEATURES)}",
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
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
