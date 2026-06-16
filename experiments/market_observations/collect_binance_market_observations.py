from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SPOT_DATA_API = "https://data-api.binance.vision"
FAPI = "https://fapi.binance.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", help="Comma-separated Binance symbols; overrides inventory")
    parser.add_argument("--quote-asset", default="USDT")
    parser.add_argument("--sample-symbols", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--skip-futures", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def utc_now_hour() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def default_run_id(timestamp: dt.datetime) -> str:
    return timestamp.strftime("%Y%m%dT%H%M%SZ")


def ms(timestamp: dt.datetime) -> int:
    return int(timestamp.timestamp() * 1000)


def fetch_json(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Any:
    query = "" if not params else "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{base_url}{path}{query}",
        headers={"User-Agent": "alpha-os-market-observations/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def rows_with_source(
    *,
    source: str,
    symbol: str,
    run_id: str,
    observed_at: str,
    rows: Iterable[Any],
) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {
            "collection_run_id": run_id,
            "observed_at": observed_at,
            "source": source,
            "symbol": symbol,
            "payload": row,
        }


def fetch_spot_exchange_info(symbol: str | None = None) -> dict[str, Any]:
    params = {"symbol": symbol} if symbol is not None else None
    return fetch_json(SPOT_DATA_API, "/api/v3/exchangeInfo", params)


def fetch_spot_symbols(quote_asset: str) -> list[str]:
    payload = fetch_spot_exchange_info()
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


def requested_symbols(
    *,
    explicit_symbols: str | None,
    quote_asset: str,
    sample_symbols: int | None,
    sample_seed: int,
) -> tuple[list[str], int, str]:
    if explicit_symbols:
        symbols = [symbol.strip().upper() for symbol in explicit_symbols.split(",") if symbol.strip()]
        if not symbols:
            raise ValueError("--symbols must contain at least one symbol when provided")
        return sorted(set(symbols)), len(symbols), "explicit"

    inventory = fetch_spot_symbols(quote_asset)
    if not inventory:
        raise RuntimeError(f"no spot symbols found for quote asset {quote_asset}")
    if sample_symbols is None or sample_symbols >= len(inventory):
        return inventory, len(inventory), "inventory"
    if sample_symbols <= 0:
        raise ValueError("--sample-symbols must be positive")
    rng = random.Random(sample_seed)
    return sorted(rng.sample(inventory, sample_symbols)), len(inventory), "inventory_sample"


def spot_kline_rows(symbol: str, interval: str, start: dt.datetime, end: dt.datetime, limit: int):
    return fetch_json(
        SPOT_DATA_API,
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": ms(start),
            "endTime": ms(end),
            "limit": limit,
        },
    )


def spot_agg_trade_rows(symbol: str, start: dt.datetime, end: dt.datetime, limit: int):
    return fetch_json(
        SPOT_DATA_API,
        "/api/v3/aggTrades",
        {
            "symbol": symbol,
            "startTime": ms(start),
            "endTime": ms(end),
            "limit": limit,
        },
    )


def spot_depth_row(symbol: str):
    return fetch_json(SPOT_DATA_API, "/api/v3/depth", {"symbol": symbol, "limit": 100})


def spot_exchange_info_row(symbol: str):
    return fetch_spot_exchange_info(symbol)


def futures_kline_rows(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    limit: int,
):
    return fetch_json(
        FAPI,
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": ms(start),
            "endTime": ms(end),
            "limit": limit,
        },
    )


def premium_index_kline_rows(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    limit: int,
):
    return fetch_json(
        FAPI,
        "/fapi/v1/premiumIndexKlines",
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": ms(start),
            "endTime": ms(end),
            "limit": limit,
        },
    )


def funding_rate_rows(symbol: str, start: dt.datetime, end: dt.datetime, limit: int):
    return fetch_json(
        FAPI,
        "/fapi/v1/fundingRate",
        {
            "symbol": symbol,
            "startTime": ms(start),
            "endTime": ms(end),
            "limit": limit,
        },
    )


def open_interest_rows(symbol: str, interval: str, start: dt.datetime, end: dt.datetime, limit: int):
    return fetch_json(
        FAPI,
        "/futures/data/openInterestHist",
        {
            "symbol": symbol,
            "period": interval,
            "startTime": ms(start),
            "endTime": ms(end),
            "limit": limit,
        },
    )


def current_open_interest_row(symbol: str):
    return fetch_json(FAPI, "/fapi/v1/openInterest", {"symbol": symbol})


def futures_depth_row(symbol: str):
    return fetch_json(FAPI, "/fapi/v1/depth", {"symbol": symbol, "limit": 100})


def futures_agg_trade_rows(symbol: str, start: dt.datetime, end: dt.datetime, limit: int):
    return fetch_json(
        FAPI,
        "/fapi/v1/aggTrades",
        {
            "symbol": symbol,
            "startTime": ms(start),
            "endTime": ms(end),
            "limit": limit,
        },
    )


def main() -> None:
    args = parse_args()
    if args.days <= 0:
        raise ValueError("--days must be positive")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    symbols, symbol_inventory_count, symbol_selection = requested_symbols(
        explicit_symbols=args.symbols,
        quote_asset=args.quote_asset,
        sample_symbols=args.sample_symbols,
        sample_seed=args.sample_seed,
    )

    observed_at_dt = utc_now()
    observed_at = observed_at_dt.isoformat()
    run_id = args.run_id or default_run_id(observed_at_dt)
    end = utc_now_hour()
    start = end - dt.timedelta(days=args.days)
    run_output_dir = args.output_dir / "runs" / run_id
    if run_output_dir.exists():
        raise SystemExit(f"collection run already exists: {run_output_dir}")
    run_output_dir.mkdir(parents=True, exist_ok=False)

    counts: dict[str, int] = {}
    errors: list[str] = []
    sources = [
        (
            "spot_klines",
            lambda symbol: spot_kline_rows(symbol, args.interval, start, end, args.limit),
        ),
        (
            "spot_agg_trades",
            lambda symbol: spot_agg_trade_rows(symbol, start, end, args.limit),
        ),
        (
            "spot_depth_snapshot",
            lambda symbol: [spot_depth_row(symbol)],
        ),
        (
            "spot_exchange_info",
            lambda symbol: [spot_exchange_info_row(symbol)],
        ),
    ]
    if not args.skip_futures:
        sources.extend(
            [
                (
                    "futures_klines",
                    lambda symbol: futures_kline_rows(symbol, args.interval, start, end, args.limit),
                ),
                (
                    "premium_index_klines",
                    lambda symbol: premium_index_kline_rows(
                        symbol, args.interval, start, end, args.limit
                    ),
                ),
                (
                    "funding_rates",
                    lambda symbol: funding_rate_rows(symbol, start, end, args.limit),
                ),
                (
                    "open_interest_history",
                    lambda symbol: open_interest_rows(symbol, args.interval, start, end, args.limit),
                ),
                (
                    "futures_agg_trades",
                    lambda symbol: futures_agg_trade_rows(symbol, start, end, args.limit),
                ),
                (
                    "current_open_interest",
                    lambda symbol: [current_open_interest_row(symbol)],
                ),
                (
                    "futures_depth_snapshot",
                    lambda symbol: [futures_depth_row(symbol)],
                ),
            ]
        )

    for source, fetcher in sources:
        rows = []
        for symbol in symbols:
            try:
                payload = fetcher(symbol)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source}:{symbol}: {exc}")
                continue
            rows.extend(
                rows_with_source(
                    source=source,
                    symbol=symbol,
                    run_id=run_id,
                    observed_at=observed_at,
                    rows=payload,
                )
            )
            time.sleep(0.1)
        counts[source] = write_jsonl(run_output_dir / f"{source}.jsonl", rows)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "# Binance Market Observation Collection",
                "",
                f"- created_at: {dt.datetime.now(dt.UTC).isoformat()}",
                f"- collection_run_id: {run_id}",
                f"- observed_at: {observed_at}",
                f"- quote_asset: {args.quote_asset}",
                f"- symbol_selection: {symbol_selection}",
                f"- symbol_inventory_count: {symbol_inventory_count}",
                f"- requested_symbol_count: {len(symbols)}",
                f"- sample_symbols: {args.sample_symbols}",
                f"- sample_seed: {args.sample_seed}",
                f"- symbols: {', '.join(symbols[:50])}",
                f"- symbols_truncated: {len(symbols) > 50}",
                f"- start: {start.isoformat()}",
                f"- end: {end.isoformat()}",
                f"- interval: {args.interval}",
                f"- skip_futures: {args.skip_futures}",
                f"- output_dir: {args.output_dir}",
                f"- run_output_dir: {run_output_dir}",
                "",
                "## Row Counts",
                "",
                *[f"- {source}: {count}" for source, count in counts.items()],
                "",
                "## Errors",
                "",
                *([f"- {error}" for error in errors] if errors else ["none"]),
                "",
                "## Guard",
                "",
                "This collector writes raw-ish public market observations.",
                "It does not choose alpha signals, build features, or define a strategy.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {run_output_dir}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
