from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "datasets" / "ds_crypto_btc_eth_daily_2024_2025"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ASSETS = ("BTCUSDT", "ETHUSDT")
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2025-12-31"
OUTPUT_COLUMNS = ("timestamp", "close", "volume", "funding_rate", "open_interest")


def _api_key() -> str | None:
    for name in ("SIGNAL_NOISE_API_KEY", "ALPHA_OS_SIGNAL_NOISE_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _get_json(
    session: requests.Session,
    *,
    base_url: str,
    path: str,
    params: dict[str, str],
    timeout: int,
    api_key: str | None,
) -> list[dict[str, Any]]:
    headers = {}
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}", "X-API-Key": api_key}
    response = session.get(
        f"{base_url.rstrip('/')}{path}",
        params=params,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise TypeError(f"expected list payload from {path}, got {type(payload).__name__}")
    return payload


def _fetch_observation(
    session: requests.Session,
    *,
    base_url: str,
    asset: str,
    observable_id: str,
    start_date: str,
    timeout: int,
    api_key: str | None,
) -> pd.DataFrame:
    payload = _get_json(
        session,
        base_url=base_url,
        path="/observations/data",
        params={
            "asset": asset,
            "observable_id": observable_id,
            "resolution": "1d",
            "source_id": "signal_noise",
            "since": start_date,
        },
        timeout=timeout,
        api_key=api_key,
    )
    if not payload:
        raise ValueError(f"signal-noise returned no rows for {asset} {observable_id}")
    frame = pd.DataFrame(payload)
    if "date" in frame.columns and "timestamp" not in frame.columns:
        frame = frame.rename(columns={"date": "timestamp"})
    if "timestamp" not in frame.columns:
        raise ValueError(f"{asset} {observable_id} response has no timestamp column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    frame["timestamp"] = frame["timestamp"].dt.floor("D")
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _value_column(frame: pd.DataFrame, *, preferred: str, observable_id: str) -> pd.Series:
    if preferred in frame.columns:
        return pd.to_numeric(frame[preferred], errors="raise")
    if "value" in frame.columns:
        return pd.to_numeric(frame["value"], errors="raise")
    raise ValueError(f"{observable_id} response has neither {preferred!r} nor 'value'")


def _build_asset_frame(
    session: requests.Session,
    *,
    base_url: str,
    asset: str,
    start_date: str,
    end_date: str,
    timeout: int,
    api_key: str | None,
) -> pd.DataFrame:
    close = _fetch_observation(
        session,
        base_url=base_url,
        asset=asset,
        observable_id="daily_close",
        start_date=start_date,
        timeout=timeout,
        api_key=api_key,
    )
    funding = _fetch_observation(
        session,
        base_url=base_url,
        asset=asset,
        observable_id="funding_rate",
        start_date=start_date,
        timeout=timeout,
        api_key=api_key,
    )
    open_interest = _fetch_observation(
        session,
        base_url=base_url,
        asset=asset,
        observable_id="open_interest",
        start_date=start_date,
        timeout=timeout,
        api_key=api_key,
    )

    frame = pd.DataFrame({"timestamp": pd.date_range(start_date, end_date, freq="D", tz="UTC")})
    frame = frame.merge(
        close.assign(
            close=_value_column(close, preferred="close", observable_id="daily_close"),
            volume=_value_column(close, preferred="volume", observable_id="daily_close"),
        )[["timestamp", "close", "volume"]],
        on="timestamp",
        how="left",
    )
    frame = frame.merge(
        funding.assign(
            funding_rate=_value_column(
                funding,
                preferred="funding_rate",
                observable_id="funding_rate",
            )
        )[["timestamp", "funding_rate"]],
        on="timestamp",
        how="left",
    )
    frame = frame.merge(
        open_interest.assign(
            open_interest=_value_column(
                open_interest,
                preferred="open_interest",
                observable_id="open_interest",
            )
        )[["timestamp", "open_interest"]],
        on="timestamp",
        how="left",
    )

    missing = frame[list(OUTPUT_COLUMNS[1:])].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        detail = ", ".join(f"{column}={count}" for column, count in missing.items())
        raise ValueError(f"{asset} dataset has missing required values: {detail}")

    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT00:00:00+00:00")
    return frame.loc[:, OUTPUT_COLUMNS]


def _write_asset(frame: pd.DataFrame, *, output_dir: Path, asset: str, dry_run: bool) -> None:
    path = output_dir / f"{asset}.csv"
    if dry_run:
        print(f"{asset}: {len(frame)} rows ready for {path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"{asset}: wrote {len(frame)} rows to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the crypto regime momentum CSV snapshot from signal-noise."
    )
    parser.add_argument("--base-url", default=os.getenv("SIGNAL_NOISE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--asset", action="append", choices=DEFAULT_ASSETS)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = tuple(args.asset) if args.asset else DEFAULT_ASSETS
    api_key = _api_key()
    with requests.Session() as session:
        for asset in assets:
            frame = _build_asset_frame(
                session,
                base_url=str(args.base_url),
                asset=asset,
                start_date=str(args.start_date),
                end_date=str(args.end_date),
                timeout=int(args.timeout),
                api_key=api_key,
            )
            _write_asset(frame, output_dir=args.output_dir, asset=asset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
