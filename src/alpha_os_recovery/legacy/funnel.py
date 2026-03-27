from __future__ import annotations

import sqlite3

from ..config import asset_data_dir
from ..evolution.discovery_pool import DiscoveryPool
from .funnel_summary import FunnelSummary, summarize_funnel


def load_funnel_summary(asset: str) -> FunnelSummary:
    asset_dir = asset_data_dir(asset)
    pool = DiscoveryPool.load_from_db(asset_dir / "discovery_pool.db")
    conn = sqlite3.connect(str(asset_dir / "alpha_registry.db"))
    conn.row_factory = sqlite3.Row
    try:
        candidate_rows = conn.execute(
            "SELECT source, status, behavior_json, error_message FROM candidates"
        ).fetchall()
        managed_rows = conn.execute(
            "SELECT state FROM alphas"
        ).fetchall()
        deployed_total = conn.execute(
            "SELECT COUNT(*) FROM deployed_alphas"
        ).fetchone()[0]
    finally:
        conn.close()

    return summarize_funnel(
        asset=asset,
        discovery_pool_entries=pool.size,
        candidate_rows=[dict(row) for row in candidate_rows],
        managed_rows=[dict(row) for row in managed_rows],
        deployed_total=deployed_total,
    )
