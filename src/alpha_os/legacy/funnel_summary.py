from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SourceFunnelSummary:
    source: str
    total: int
    pending: int
    validating: int
    adopted: int
    rejected: int
    reject_axes: list[tuple[str, int]]
    top_reject_reasons: list[tuple[str, int]]


@dataclass(frozen=True)
class FunnelSummary:
    asset: str
    discovery_pool_entries: int
    candidate_total: int
    candidate_pending: int
    candidate_validating: int
    candidate_adopted: int
    candidate_rejected: int
    enqueued_total: int
    enqueued_manual: int
    managed_candidate: int
    managed_active: int
    managed_dormant: int
    managed_rejected: int
    deployed_total: int
    reject_axes: list[tuple[str, int]]
    reject_reasons: list[tuple[str, int]]
    source_summaries: list[SourceFunnelSummary]


def reject_axis(reason: str) -> str:
    prefix, sep, _rest = reason.partition(":")
    if sep and prefix in {"quality", "diversity", "confidence", "deployability"}:
        return prefix
    return "uncategorized"


def summarize_funnel(
    *,
    asset: str,
    discovery_pool_entries: int,
    candidate_rows: list[Mapping[str, object]],
    managed_rows: list[Mapping[str, object]],
    deployed_total: int,
) -> FunnelSummary:
    candidate_pending = 0
    candidate_validating = 0
    candidate_adopted = 0
    candidate_rejected = 0
    enqueued_total = 0
    enqueued_manual = 0
    reject_axis_counts: dict[str, int] = {}
    reject_reason_counts: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}
    source_reject_axes: dict[str, dict[str, int]] = {}
    source_reject_reasons: dict[str, dict[str, int]] = {}

    for row in candidate_rows:
        source = str(row.get("source") or "").strip() or "<legacy>"
        status = str(row["status"])
        if status == "pending":
            candidate_pending += 1
        elif status == "validating":
            candidate_validating += 1
        elif status == "adopted":
            candidate_adopted += 1
        elif status == "rejected":
            candidate_rejected += 1

        behavior_json = str(row.get("behavior_json") or "{}")
        try:
            behavior = json.loads(behavior_json)
        except json.JSONDecodeError:
            behavior = {}
        behavior_source = behavior.get("source", "")
        if source.startswith("alpha_generator_") or behavior_source == "alpha_generator":
            enqueued_total += 1
            if behavior.get("enqueue") == "manual_discovery_pool":
                enqueued_manual += 1

        counts = source_counts.setdefault(
            source,
            {"total": 0, "pending": 0, "validating": 0, "adopted": 0, "rejected": 0},
        )
        counts["total"] += 1
        if status in counts:
            counts[status] += 1

        reason = str(row.get("error_message") or "").strip()
        if reason:
            axis = reject_axis(reason)
            reject_axis_counts[axis] = reject_axis_counts.get(axis, 0) + 1
            reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1
            axes = source_reject_axes.setdefault(source, {})
            axes[axis] = axes.get(axis, 0) + 1
            reasons = source_reject_reasons.setdefault(source, {})
            reasons[reason] = reasons.get(reason, 0) + 1

    managed_candidate = 0
    managed_active = 0
    managed_dormant = 0
    managed_rejected = 0
    for row in managed_rows:
        state = str(row["state"])
        if state == "candidate":
            managed_candidate += 1
        elif state == "active":
            managed_active += 1
        elif state == "dormant":
            managed_dormant += 1
        elif state == "rejected":
            managed_rejected += 1

    source_summaries = [
        SourceFunnelSummary(
            source=source,
            total=counts["total"],
            pending=counts["pending"],
            validating=counts["validating"],
            adopted=counts["adopted"],
            rejected=counts["rejected"],
            reject_axes=sorted(
                source_reject_axes.get(source, {}).items(),
                key=lambda item: (-item[1], item[0]),
            ),
            top_reject_reasons=sorted(
                source_reject_reasons.get(source, {}).items(),
                key=lambda item: (-item[1], item[0]),
            )[:5],
        )
        for source, counts in sorted(
            source_counts.items(),
            key=lambda item: (
                0 if item[0].startswith("alpha_generator_") else 1 if item[0] == "manual" else 2,
                item[0],
            ),
        )
    ]

    return FunnelSummary(
        asset=asset,
        discovery_pool_entries=discovery_pool_entries,
        candidate_total=len(candidate_rows),
        candidate_pending=candidate_pending,
        candidate_validating=candidate_validating,
        candidate_adopted=candidate_adopted,
        candidate_rejected=candidate_rejected,
        enqueued_total=enqueued_total,
        enqueued_manual=enqueued_manual,
        managed_candidate=managed_candidate,
        managed_active=managed_active,
        managed_dormant=managed_dormant,
        managed_rejected=managed_rejected,
        deployed_total=deployed_total,
        reject_axes=sorted(
            reject_axis_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ),
        reject_reasons=sorted(
            reject_reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:10],
        source_summaries=source_summaries,
    )
