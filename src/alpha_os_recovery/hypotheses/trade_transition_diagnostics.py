from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .batch_research_diagnostics import is_batch_research_record
from .identity import expression_feature_families


@dataclass(frozen=True)
class BatchTransitionState:
    hypothesis_id: str
    family_label: str
    stake: float
    blended_quality: float
    live_quality: float
    confidence: float
    signal_ratio: float
    signal_mean_abs: float
    research_retained: bool
    live_proven: bool
    actionable_live: bool
    capital_eligible: bool
    capital_backed: bool
    capital_reason: str
    live_promotion_blocker: str
    redundancy_capped_by: str
    research_candidate_capped: bool


def capture_batch_transition_snapshot(
    records,
    *,
    families: tuple[str, ...] | None = None,
) -> dict[str, BatchTransitionState]:
    family_filter = set(families or ())
    captured: dict[str, BatchTransitionState] = {}
    for record in records:
        if not is_batch_research_record(record):
            continue
        record_families = set(expression_feature_families(record.expression))
        if family_filter and not (record_families & family_filter):
            continue
        metadata = getattr(record, "metadata", {}) or {}
        captured[record.hypothesis_id] = BatchTransitionState(
            hypothesis_id=record.hypothesis_id,
            family_label=",".join(sorted(record_families)) or "unknown",
            stake=float(record.stake),
            blended_quality=float(metadata.get("lifecycle_blended_quality", 0.0) or 0.0),
            live_quality=float(metadata.get("lifecycle_live_quality", 0.0) or 0.0),
            confidence=float(metadata.get("lifecycle_quality_confidence", 0.0) or 0.0),
            signal_ratio=float(metadata.get("lifecycle_signal_nonzero_ratio", 0.0) or 0.0),
            signal_mean_abs=float(metadata.get("lifecycle_signal_mean_abs", 0.0) or 0.0),
            research_retained=bool(metadata.get("lifecycle_research_retained", False)),
            live_proven=bool(metadata.get("lifecycle_live_proven", False)),
            actionable_live=bool(metadata.get("lifecycle_actionable_live", False)),
            capital_eligible=bool(metadata.get("lifecycle_capital_eligible", False)),
            capital_backed=bool(
                metadata.get(
                    "lifecycle_capital_backed",
                    bool(metadata.get("lifecycle_capital_eligible", False))
                    and float(record.stake) > 0,
                )
            ),
            capital_reason=str(metadata.get("lifecycle_capital_reason", "")),
            live_promotion_blocker=str(metadata.get("lifecycle_live_promotion_blocker", "")),
            redundancy_capped_by=str(metadata.get("lifecycle_redundancy_capped_by", "")),
            research_candidate_capped=bool(
                metadata.get("lifecycle_research_candidate_capped", False)
            ),
        )
    return captured


def batch_transition_drop_reason(state: BatchTransitionState) -> str:
    if state.capital_backed:
        return "backed"
    if state.redundancy_capped_by:
        return "redundancy"
    if state.research_candidate_capped:
        return "candidate_cap"
    if not state.research_retained:
        return "research_q"
    blocker = state.live_promotion_blocker
    if blocker == "insufficient_observations":
        return "obs"
    if blocker == "weak_signal_activity":
        return "signal"
    if blocker in {"weak_marginal_contribution", "weak_live_quality_and_contribution"}:
        return "contrib"
    if blocker == "weak_live_quality":
        return "live_q"
    if not state.actionable_live:
        return "not_actionable"
    return state.capital_reason or "other"


def build_trade_transition_summary(
    pre_snapshot: dict[str, BatchTransitionState],
    post_snapshot: dict[str, BatchTransitionState],
    *,
    top: int = 5,
) -> dict[str, object]:
    pre_ids = set(pre_snapshot)
    post_ids = set(post_snapshot)
    all_ids = pre_ids | post_ids

    entries: list[str] = []
    exits: list[str] = []
    exit_reasons: Counter[str] = Counter()
    entry_reasons: Counter[str] = Counter()
    changed: list[tuple[float, str]] = []

    pre_backed = sum(1 for state in pre_snapshot.values() if state.capital_backed)
    post_backed = sum(1 for state in post_snapshot.values() if state.capital_backed)

    for hypothesis_id in all_ids:
        pre = pre_snapshot.get(hypothesis_id)
        post = post_snapshot.get(hypothesis_id)
        pre_backed_now = pre is not None and pre.capital_backed
        post_backed_now = post is not None and post.capital_backed
        if not pre_backed_now and post_backed_now:
            entries.append(hypothesis_id)
            if post is not None:
                entry_reasons[post.capital_reason or "backed"] += 1
        elif pre_backed_now and not post_backed_now:
            exits.append(hypothesis_id)
            if post is not None:
                exit_reasons[batch_transition_drop_reason(post)] += 1
        if pre is not None and post is not None and pre.stake != post.stake:
            changed.append((abs(post.stake - pre.stake), hypothesis_id))

    changed.sort(reverse=True)

    def _format_entry(hypothesis_id: str) -> str:
        pre = pre_snapshot.get(hypothesis_id)
        post = post_snapshot.get(hypothesis_id)
        if post is None:
            return hypothesis_id
        pre_stake = 0.0 if pre is None else pre.stake
        return (
            f"{hypothesis_id} fam={post.family_label} "
            f"{pre_stake:.3f}->{post.stake:.3f} "
            f"q={post.blended_quality:.2f} live_q={post.live_quality:.2f} "
            f"sig={post.signal_ratio:.2f}/{post.signal_mean_abs:.2f} "
            f"reason={post.capital_reason or 'backed'}"
        )

    def _format_exit(hypothesis_id: str) -> str:
        pre = pre_snapshot.get(hypothesis_id)
        post = post_snapshot.get(hypothesis_id)
        if pre is None:
            return hypothesis_id
        if post is None:
            return (
                f"{hypothesis_id} fam={pre.family_label} "
                f"{pre.stake:.3f}->missing q={pre.blended_quality:.2f}"
            )
        return (
            f"{hypothesis_id} fam={post.family_label} "
            f"{pre.stake:.3f}->{post.stake:.3f} "
            f"q={post.blended_quality:.2f} live_q={post.live_quality:.2f} "
            f"sig={post.signal_ratio:.2f}/{post.signal_mean_abs:.2f} "
            f"reason={batch_transition_drop_reason(post)}"
        )

    return {
        "scoped_pre": len(pre_snapshot),
        "scoped_post": len(post_snapshot),
        "pre_backed": pre_backed,
        "post_backed": post_backed,
        "entered": len(entries),
        "exited": len(exits),
        "entry_reasons": entry_reasons,
        "exit_reasons": exit_reasons,
        "top_entries": [_format_entry(hid) for hid in entries[: max(top, 0)]],
        "top_exits": [_format_exit(hid) for hid in exits[: max(top, 0)]],
        "top_changed": [
            _format_entry(hid) if hid in entries else _format_exit(hid)
            for _, hid in changed[: max(top, 0)]
        ],
    }
