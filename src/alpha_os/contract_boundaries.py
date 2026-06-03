from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractFieldGroup:
    group_name: str
    field_paths: tuple[str, ...]

    def contains(self, field_path: str) -> bool:
        return field_path in self.field_paths


@dataclass(frozen=True)
class SubjectSetContractBoundary:
    field_groups: tuple[ContractFieldGroup, ...]

    def group_for_field(self, field_path: str) -> str | None:
        for group in self.field_groups:
            if group.contains(field_path):
                return group.group_name
        return None


@dataclass(frozen=True)
class PortfolioConstraintBoundary:
    sizing_time_fields: tuple[str, ...]
    post_sizing_normalization_fields: tuple[str, ...]

    def is_sizing_time(self, field_name: str) -> bool:
        return field_name in self.sizing_time_fields

    def is_post_sizing_normalization(self, field_name: str) -> bool:
        return field_name in self.post_sizing_normalization_fields

    def stage_for_field(self, field_name: str) -> str | None:
        if self.is_sizing_time(field_name):
            return "sizing_time"
        if self.is_post_sizing_normalization(field_name):
            return "post_sizing_normalization"
        return None


def default_subject_set_contract_boundary() -> SubjectSetContractBoundary:
    return SubjectSetContractBoundary(
        field_groups=(
            ContractFieldGroup(
                group_name="subject",
                field_paths=(
                    "subject.subject_id",
                    "subject.asset",
                    "subject.subject_kind",
                ),
            ),
        )
    )


def default_portfolio_constraint_boundary() -> PortfolioConstraintBoundary:
    return PortfolioConstraintBoundary(
        sizing_time_fields=("target_vol",),
        post_sizing_normalization_fields=(
            "direction_mode",
            "gross_exposure_cap",
            "gross_leverage_cap",
            "net_exposure_target",
            "asset_class_weight_caps",
            "cluster_weight_caps",
        ),
    )


def subject_set_contract_groups(
    boundary: SubjectSetContractBoundary,
) -> tuple[str, ...]:
    return tuple(group.group_name for group in boundary.field_groups)


def format_subject_set_contract_groups(
    boundary: SubjectSetContractBoundary,
) -> str:
    return ",".join(subject_set_contract_groups(boundary))


def active_constraint_stages(
    boundary: PortfolioConstraintBoundary,
    *,
    field_values: dict[str, object | None],
) -> tuple[str, ...]:
    def is_present(value: object | None) -> bool:
        if value is None:
            return False
        if isinstance(value, (dict, tuple, list, set)):
            return len(value) > 0
        return True

    active_sizing_fields = tuple(
        field_name
        for field_name in boundary.sizing_time_fields
        if is_present(field_values.get(field_name))
    )
    active_post_sizing_fields = tuple(
        field_name
        for field_name in boundary.post_sizing_normalization_fields
        if is_present(field_values.get(field_name))
    )
    parts: list[str] = []
    if active_sizing_fields:
        parts.append("sizing_time:" + ",".join(active_sizing_fields))
    if active_post_sizing_fields:
        parts.append(
            "post_sizing_normalization:" + ",".join(active_post_sizing_fields)
        )
    return tuple(parts)


def format_active_constraint_stages(
    boundary: PortfolioConstraintBoundary,
    *,
    field_values: dict[str, object | None],
) -> str:
    return ";".join(active_constraint_stages(boundary, field_values=field_values))
