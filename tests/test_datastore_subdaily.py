"""Tests for DataStore subdaily (resolution) support."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from alpha_os.data.store import DataStore


@pytest.fixture
def store(tmp_path):
    s = DataStore(tmp_path / "test.db")
    yield s
    s.close()


class TestMigration:
    def test_resolution_column_exists(self, store):
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(signals)")}
        assert "resolution" in cols

    def test_migration_idempotent(self, tmp_path):
        s = DataStore(tmp_path / "test2.db")
        # Second open should not fail
        s2 = DataStore(tmp_path / "test2.db")
        cols = {r[1] for r in s2._conn.execute("PRAGMA table_info(signals)")}
        assert "resolution" in cols
        s.close()
        s2.close()


class TestSyncHourly:
    def test_sync_hourly(self, tmp_path):
        mock_client = MagicMock()
        mock_client.stale_signals.return_value = []
        timestamps = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="h")
        mock_client.get_batch.return_value = {
            "funding_rate_btc": pd.DataFrame({
                "timestamp": timestamps,
                "value": [0.0001, 0.0002, 0.0003, 0.0004],
            }),
        }

        store = DataStore(tmp_path / "sync.db", client=mock_client)
        store.sync(["funding_rate_btc"], resolution="1h")

        rows = store._conn.execute(
            "SELECT date, value, resolution FROM signals WHERE name = 'funding_rate_btc'"
            " ORDER BY date"
        ).fetchall()
        assert len(rows) == 4
        assert rows[0][2] == "1h"
        # Hourly timestamps should include time, not just date
        assert "00:00" in str(rows[0][0])
        store.close()

    def test_sync_daily_backward_compatible(self, tmp_path):
        mock_client = MagicMock()
        mock_client.stale_signals.return_value = []
        timestamps = pd.date_range("2024-01-01", periods=3, freq="D")
        mock_client.get_batch.return_value = {
            "btc_ohlcv": pd.DataFrame({
                "timestamp": timestamps,
                "value": [42000.0, 43000.0, 44000.0],
            }),
        }

        store = DataStore(tmp_path / "sync.db", client=mock_client)
        store.sync(["btc_ohlcv"])  # default resolution="1d"

        rows = store._conn.execute(
            "SELECT date, resolution FROM signals WHERE name = 'btc_ohlcv' ORDER BY date"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0][1] == "1d"
        # Daily should be date-only format
        assert len(rows[0][0]) == 10
        store.close()


class TestGetMatrixHourly:
    def test_get_matrix_hourly(self, store):
        # Insert hourly data
        for i in range(4):
            store._conn.execute(
                "INSERT INTO signals (name, date, value, resolution)"
                " VALUES (?, ?, ?, ?)",
                ("sig_a", f"2024-01-01T{i:02d}:00:00", float(i), "1h"),
            )
            store._conn.execute(
                "INSERT INTO signals (name, date, value, resolution)"
                " VALUES (?, ?, ?, ?)",
                ("sig_b", f"2024-01-01T{i:02d}:00:00", float(i * 10), "1h"),
            )
        store._conn.commit()

        matrix = store.get_matrix(["sig_a", "sig_b"], resolution="1h")
        assert len(matrix) == 4
        assert "sig_a" in matrix.columns
        assert "sig_b" in matrix.columns
        assert matrix["sig_a"].iloc[3] == 3.0
        assert matrix["sig_b"].iloc[3] == 30.0

    def test_get_matrix_daily_default(self, store):
        # Insert daily data (no explicit resolution → defaults to '1d')
        for i in range(3):
            store._conn.execute(
                "INSERT INTO signals (name, date, value)"
                " VALUES (?, ?, ?)",
                ("daily_sig", f"2024-01-{i+1:02d}", float(i)),
            )
        store._conn.commit()

        matrix = store.get_matrix(["daily_sig"])
        assert len(matrix) == 3

    def test_resolution_isolation(self, store):
        # Same signal name, different resolutions
        store._conn.execute(
            "INSERT INTO signals (name, date, value, resolution)"
            " VALUES ('sig', '2024-01-01', 100.0, '1d')"
        )
        store._conn.execute(
            "INSERT INTO signals (name, date, value, resolution)"
            " VALUES ('sig', '2024-01-01T00:00:00', 50.0, '1h')"
        )
        store._conn.execute(
            "INSERT INTO signals (name, date, value, resolution)"
            " VALUES ('sig', '2024-01-01T01:00:00', 60.0, '1h')"
        )
        store._conn.commit()

        daily = store.get_matrix(["sig"], resolution="1d")
        hourly = store.get_matrix(["sig"], resolution="1h")

        assert len(daily) == 1
        assert daily["sig"].iloc[0] == 100.0
        assert len(hourly) == 2
        assert hourly["sig"].iloc[0] == 50.0


class TestBackwardCompatible:
    def test_existing_daily_data_works(self, store):
        # Simulate pre-migration data (resolution column defaults to '1d')
        for i in range(5):
            store._conn.execute(
                "INSERT INTO signals (name, date, value)"
                " VALUES (?, ?, ?)",
                ("old_sig", f"2024-01-{i+1:02d}", float(i)),
            )
        store._conn.commit()

        # get_matrix without resolution should still work
        matrix = store.get_matrix(["old_sig"])
        assert len(matrix) == 5
        assert matrix["old_sig"].iloc[0] == 0.0

    def test_sync_default_resolution(self, tmp_path):
        mock_client = MagicMock()
        mock_client.stale_signals.return_value = []
        mock_client.get_batch.return_value = {}

        store = DataStore(tmp_path / "bc.db", client=mock_client)
        store.sync(["sig1"])  # no resolution arg → "1d"

        # Should call get_batch with resolution=None (daily default)
        mock_client.get_batch.assert_called_once()
        call_kwargs = mock_client.get_batch.call_args
        assert call_kwargs[1].get("resolution") is None or call_kwargs[1].get("resolution") is None
        store.close()
