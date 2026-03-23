"""Compatibility wrapper for legacy admission replay helpers."""

from alpha_os.legacy.admission_replay import (
    RegistryRebuildStats,
    alpha_id_for_expression,
    apply_registry_snapshot,
    backup_registry_db,
    existing_alpha_ids_by_expression,
    load_candidate_records,
    load_registry_records,
    load_source_records,
    materialize_admission_snapshot,
    rebuild_registry,
)

__all__ = [
    "RegistryRebuildStats",
    "alpha_id_for_expression",
    "apply_registry_snapshot",
    "backup_registry_db",
    "existing_alpha_ids_by_expression",
    "load_candidate_records",
    "load_registry_records",
    "load_source_records",
    "materialize_admission_snapshot",
    "rebuild_registry",
]
