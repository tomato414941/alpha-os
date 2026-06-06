from __future__ import annotations

import sys
from datetime import date
from pathlib import Path


def test_fetch_binance_spot_daily_rows_parses_klines(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from strategies.crypto_momentum import fetch_market_data

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[list[str | int]]:
            return [
                [
                    1704067200000,
                    "0",
                    "0",
                    "0",
                    "44167.33203125",
                    "18426978443.0",
                ]
            ]

    def fake_get(*args, **kwargs):
        return Response()

    monkeypatch.setattr(fetch_market_data.requests, "get", fake_get)

    rows = fetch_market_data.fetch_binance_spot_daily_rows(
        "BTCUSDT",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
    )

    assert rows[0].timestamp == "2024-01-01T00:00:00+00:00"
    assert rows[0].close == 44167.33203125
    assert rows[0].volume == 18426978443.0


def test_write_daily_rows_uses_existing_crypto_momentum_csv_shape(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from strategies.crypto_momentum.fetch_market_data import (
        BinanceDailyRow,
        write_daily_rows,
    )

    path = write_daily_rows(
        "BTCUSDT",
        (
            BinanceDailyRow(
                timestamp="2024-01-01T00:00:00+00:00",
                close=44167.33203125,
                volume=18426978443.0,
            ),
        ),
        output_dir=tmp_path,
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        "timestamp,close,volume,funding_rate,open_interest",
        "2024-01-01T00:00:00+00:00,44167.33203125,18426978443.0,,",
    ]
