from __future__ import annotations

from collections.abc import Iterable

DEFAULT_REFERENCE_ASSET = "BTC"
DEFAULT_SCOPE = {"asset": DEFAULT_REFERENCE_ASSET, "universe": "core_universe_1000"}


def scope_asset(scope: dict | None, *, default: str = DEFAULT_REFERENCE_ASSET) -> str:
    if not isinstance(scope, dict):
        return default.upper()
    raw = scope.get("asset") or default
    return str(raw).upper()


def with_scope_asset(
    scope: dict | None,
    asset: str,
    *,
    default_universe: str = "core_universe_1000",
) -> dict:
    next_scope = dict(scope or {})
    next_scope["asset"] = str(asset).upper()
    next_scope.setdefault("universe", default_universe)
    return next_scope


def record_in_asset_sleeve(
    record,
    asset: str,
    *,
    default: str = DEFAULT_REFERENCE_ASSET,
) -> bool:
    return scope_asset(getattr(record, "scope", None), default=default) == str(asset).upper()


def filter_records_by_assets(
    records: Iterable,
    assets: Iterable[str] | None,
    *,
    default: str = DEFAULT_REFERENCE_ASSET,
):
    if not assets:
        return list(records)
    allowed = {str(asset).upper() for asset in assets}
    return [
        record
        for record in records
        if scope_asset(getattr(record, "scope", None), default=default) in allowed
    ]
