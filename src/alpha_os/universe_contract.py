from __future__ import annotations


def validate_subject_set_universe_contract(subject_set) -> None:
    if len(subject_set.bindings) <= 1:
        return
    universe_policy = getattr(subject_set, "universe_policy", None)
    if universe_policy is None:
        raise ValueError(
            "subject set is missing universe policy for multi-subject validation/evaluation: "
            f"{subject_set.subject_set_id}"
        )
    missing: list[str] = []
    if universe_policy.base_currency is None:
        missing.append("base_currency")
    if universe_policy.trading_calendar is None:
        missing.append("trading_calendar")
    if universe_policy.benchmark_id is None:
        missing.append("benchmark_id")
    if missing:
        raise ValueError(
            "subject set universe policy is incomplete for multi-subject validation/evaluation: "
            f"{subject_set.subject_set_id} missing {', '.join(missing)}"
        )
