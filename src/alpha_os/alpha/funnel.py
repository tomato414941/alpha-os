from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from ..config import asset_data_dir
from ..evolution.discovery_pool import DiscoveryPool


@dataclass(frozen=True)
class FunnelSummary:
    asset: str
    discovery_pool_entries: int
    candidate_total: int
    candidate_pending: int
    candidate_validating: int
    candidate_adopted: int
    candidate_rejected: int
    promoted_total: int
    promoted_manual: int
    managed_candidate: int
    managed_active: int
    managed_dormant: int
    managed_rejected: int
    deployed_total: int
    reject_reasons: list[tuple[str, int]]


def load_funnel_summary(asset: str) -> FunnelSummary:
    asset_dir = asset_data_dir(asset)
    pool = DiscoveryPool.load_from_db(asset_dir / "archive.db")
    conn = sqlite3.connect(str(asset_dir / "alpha_registry.db"))
    conn.row_factory = sqlite3.Row
    try:
        candidate_rows = conn.execute(
            "SELECT status, behavior_json, error_message FROM candidates"
        ).fetchall()
        managed_rows = conn.execute(
            "SELECT state FROM alphas"
        ).fetchall()
        deployed_total = conn.execute(
            "SELECT COUNT(*) FROM deployed_alphas"
        ).fetchone()[0]
    finally:
        conn.close()

    candidate_pending = 0
    candidate_validating = 0
    candidate_adopted = 0
    candidate_rejected = 0
    promoted_total = 0
    promoted_manual = 0
    reject_reason_counts: dict[str, int] = {}

    for row in candidate_rows:
        status = row["status"]
        if status == "pending":
            candidate_pending += 1
        elif status == "validating":
            candidate_validating += 1
        elif status == "adopted":
            candidate_adopted += 1
        elif status == "rejected":
            candidate_rejected += 1

        behavior_json = row["behavior_json"] or "{}"
        try:
            behavior = json.loads(behavior_json)
        except json.JSONDecodeError:
            behavior = {}
        source = behavior.get("source", "")
        if source == "alpha_generator":
            promoted_total += 1
            if behavior.get("promotion") == "manual_discovery_pool":
                promoted_manual += 1

        reason = (row["error_message"] or "").strip()
        if reason:
            reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1

    managed_candidate = 0
    managed_active = 0
    managed_dormant = 0
    managed_rejected = 0
    for row in managed_rows:
        state = row["state"]
        if state == "candidate":
            managed_candidate += 1
        elif state == "active":
            managed_active += 1
        elif state == "dormant":
            managed_dormant += 1
        elif state == "rejected":
            managed_rejected += 1

    top_rejects = sorted(
        reject_reason_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]

    return FunnelSummary(
        asset=asset,
        discovery_pool_entries=pool.size,
        candidate_total=len(candidate_rows),
        candidate_pending=candidate_pending,
        candidate_validating=candidate_validating,
        candidate_adopted=candidate_adopted,
        candidate_rejected=candidate_rejected,
        promoted_total=promoted_total,
        promoted_manual=promoted_manual,
        managed_candidate=managed_candidate,
        managed_active=managed_active,
        managed_dormant=managed_dormant,
        managed_rejected=managed_rejected,
        deployed_total=deployed_total,
        reject_reasons=top_rejects,
    )
