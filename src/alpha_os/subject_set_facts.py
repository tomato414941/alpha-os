from __future__ import annotations

from .contract_boundaries import format_subject_set_contract_groups


def format_subject_set_facts(definition) -> str:
    instrument_types = sorted({item.instrument_type for item in definition.instruments})
    asset_classes = sorted(
        {item.asset_class for item in definition.instruments if item.asset_class is not None}
    )
    regions = sorted(
        {item.region for item in definition.instruments if item.region is not None}
    )
    clusters = sorted(
        {item.cluster for item in definition.instruments if item.cluster is not None}
    )
    subject_kinds = sorted({item.subject_kind for item in definition.bindings})
    parts = [
        f"bindings={len(definition.bindings)}",
        f"instruments={len(definition.instruments)}",
        "subject_kinds=" + ("-" if not subject_kinds else ",".join(subject_kinds)),
        "instrument_types=" + ("-" if not instrument_types else ",".join(instrument_types)),
    ]
    if asset_classes:
        parts.append("asset_classes=" + ",".join(asset_classes))
    if regions:
        parts.append("regions=" + ",".join(regions))
    if clusters:
        parts.append("clusters=" + ",".join(clusters))
    universe_policy = getattr(definition, "universe_policy", None)
    if universe_policy is not None:
        if universe_policy.base_currency:
            parts.append("base_currency=" + universe_policy.base_currency)
        if universe_policy.trading_calendar:
            parts.append("trading_calendar=" + universe_policy.trading_calendar)
        if universe_policy.benchmark_id:
            parts.append("benchmark_id=" + universe_policy.benchmark_id)
    contract_boundary = getattr(definition, "contract_boundary", None)
    if contract_boundary is not None:
        parts.append(
            "contract_groups="
            + format_subject_set_contract_groups(contract_boundary)
        )
    return " ".join(parts)
