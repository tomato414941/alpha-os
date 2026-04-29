from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feature_plane import PriceFeaturePlane


@dataclass(frozen=True)
class RollResolution:
    contract_id: str | None
    contract_family: str | None
    quote_ccy: str | None
    collateral_ccy: str | None
    expiry: str | None
    days_to_expiry: int | None
    rolled: bool
    from_contract_id: str | None
    to_contract_id: str | None
    roll_reason: str | None

    def to_document(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "contract_family": self.contract_family,
            "quote_ccy": self.quote_ccy,
            "collateral_ccy": self.collateral_ccy,
            "expiry": self.expiry,
            "days_to_expiry": self.days_to_expiry,
            "rolled": self.rolled,
            "from_contract_id": self.from_contract_id,
            "to_contract_id": self.to_contract_id,
            "roll_reason": self.roll_reason,
        }


def _series_value(
    plane: PriceFeaturePlane,
    *,
    observable_id: str,
    date: str,
) -> str | None:
    series = plane.metadata_series(observable_id=observable_id)
    if series is None or date not in series.index:
        return None
    value = series.loc[date]
    if pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _calendar_days_to_expiry(date: str, expiry: str) -> int | None:
    current = pd.Timestamp(date)
    expiry_ts = pd.Timestamp(expiry)
    if pd.isna(current) or pd.isna(expiry_ts):
        return None
    return int((expiry_ts.normalize() - current.normalize()).days)


def _business_days_to_expiry(date: str, expiry: str) -> int | None:
    current = pd.Timestamp(date)
    expiry_ts = pd.Timestamp(expiry)
    if pd.isna(current) or pd.isna(expiry_ts):
        return None
    if expiry_ts.normalize() < current.normalize():
        return int((expiry_ts.normalize() - current.normalize()).days)
    business_days = pd.bdate_range(
        start=current.normalize(),
        end=expiry_ts.normalize(),
        inclusive="right",
    )
    return int(len(business_days))


def _parse_roll_rule(roll_rule: str | None) -> tuple[str | None, int | None]:
    if roll_rule is None:
        return None, None
    normalized = str(roll_rule).strip()
    if not normalized:
        return None, None
    if ":" not in normalized:
        return normalized, None
    rule_name, _, threshold = normalized.partition(":")
    if not threshold:
        return rule_name, None
    return rule_name, int(threshold)


def resolve_roll_resolution(
    plane: PriceFeaturePlane,
    *,
    date: str,
    roll_rule: str | None,
    contract_family: str | None,
    quote_ccy: str | None,
    collateral_ccy: str | None,
) -> RollResolution:
    contract_id = _series_value(plane, observable_id="contract_id", date=date)
    next_contract_id = _series_value(plane, observable_id="next_contract_id", date=date)
    expiry = _series_value(plane, observable_id="expiry", date=date)
    rule_name, threshold = _parse_roll_rule(roll_rule)

    days_to_expiry: int | None = None
    rolled = False
    roll_reason: str | None = None
    from_contract_id: str | None = None
    to_contract_id: str | None = None

    if rule_name == "calendar_days_before_expiry" and expiry is not None:
        days_to_expiry = _calendar_days_to_expiry(date, expiry)
        if (
            days_to_expiry is not None
            and threshold is not None
            and next_contract_id is not None
            and days_to_expiry <= threshold
        ):
            rolled = True
            from_contract_id = contract_id
            to_contract_id = next_contract_id
            roll_reason = rule_name
            contract_id = next_contract_id
    elif rule_name == "business_days_before_expiry" and expiry is not None:
        days_to_expiry = _business_days_to_expiry(date, expiry)
        if (
            days_to_expiry is not None
            and threshold is not None
            and next_contract_id is not None
            and days_to_expiry <= threshold
        ):
            rolled = True
            from_contract_id = contract_id
            to_contract_id = next_contract_id
            roll_reason = rule_name
            contract_id = next_contract_id
    elif rule_name in {
        None,
        "front_month",
        "volume_switch",
        "open_interest_switch",
        "perpetual",
    }:
        days_to_expiry = (
            None if expiry is None else _calendar_days_to_expiry(date, expiry)
        )
    else:
        raise ValueError(f"unsupported roll_rule for lifecycle resolution: {roll_rule}")

    return RollResolution(
        contract_id=contract_id,
        contract_family=contract_family,
        quote_ccy=quote_ccy,
        collateral_ccy=collateral_ccy,
        expiry=expiry,
        days_to_expiry=days_to_expiry,
        rolled=rolled,
        from_contract_id=from_contract_id,
        to_contract_id=to_contract_id,
        roll_reason=roll_reason,
    )
