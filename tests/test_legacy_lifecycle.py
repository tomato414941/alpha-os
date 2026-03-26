import sqlite3

import pytest

from alpha_os.legacy.stake_update import (
    compute_daily_marginal_contributions,
    compute_rolling_marginal_stake,
)


def test_compute_daily_marginal_contributions_returns_leave_one_out_values(tmp_path):
    db_path = tmp_path / "observations.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE hypothesis_observations (
                hypothesis_id TEXT NOT NULL,
                date TEXT NOT NULL,
                signal_value REAL NOT NULL,
                daily_return REAL NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO hypothesis_observations
                (hypothesis_id, date, signal_value, daily_return)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("h1", "2026-03-20", 1.0, 0.1),
                ("h2", "2026-03-20", -1.0, -0.1),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    marginals = compute_daily_marginal_contributions(
        str(db_path),
        ["h1", "h2"],
        {"h1": 1.0, "h2": 1.0},
        "2026-03-20",
    )

    assert marginals["h1"] == pytest.approx(0.1)
    assert marginals["h2"] == pytest.approx(-0.1)


def test_compute_rolling_marginal_stake_uses_prior_before_min_observations():
    assert compute_rolling_marginal_stake([0.1] * 5, prior_stake=0.3) == pytest.approx(0.3)


def test_compute_rolling_marginal_stake_clips_negative_mean_to_zero():
    history = [-0.4] * 12

    assert compute_rolling_marginal_stake(history, prior_stake=0.3) == pytest.approx(0.0)
