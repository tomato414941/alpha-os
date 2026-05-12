from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignalDiscoveryRun:
    signal_discovery_run_id: str
    signal_discovery_id: str
    subject_set_id: str
    target_id: str
    execution_start_date: str
    execution_end_date: str
    snapshot_set_id: str
    screening_result_id: str
    compressed_belief_id: str
    workflow_runtime_s: float
    total_executables: int
    pre_screen_selected: int
    probe_selected: int
    survivor_selected: int
    persisted_signals: int
    evaluation_inputs: int
    pruned_snapshots: int
    created_at: str

    def to_document(self) -> dict[str, Any]:
        return {
            "signal_discovery_id": self.signal_discovery_id,
            "subject_set_id": self.subject_set_id,
            "target_id": self.target_id,
            "execution_start_date": self.execution_start_date,
            "execution_end_date": self.execution_end_date,
            "snapshot_set_id": self.snapshot_set_id,
            "screening_result_id": self.screening_result_id,
            "compressed_belief_id": self.compressed_belief_id,
            "workflow_runtime_s": self.workflow_runtime_s,
            "total_executables": self.total_executables,
            "pre_screen_selected": self.pre_screen_selected,
            "probe_selected": self.probe_selected,
            "survivor_selected": self.survivor_selected,
            "persisted_signals": self.persisted_signals,
            "evaluation_inputs": self.evaluation_inputs,
            "pruned_snapshots": self.pruned_snapshots,
            "created_at": self.created_at,
        }

    @classmethod
    def from_document(
        cls,
        *,
        signal_discovery_run_id: str,
        document: dict[str, Any],
    ) -> "SignalDiscoveryRun":
        return cls(
            signal_discovery_run_id=signal_discovery_run_id,
            signal_discovery_id=str(document["signal_discovery_id"]),
            subject_set_id=str(document["subject_set_id"]),
            target_id=str(document["target_id"]),
            execution_start_date=str(document["execution_start_date"]),
            execution_end_date=str(document["execution_end_date"]),
            snapshot_set_id=str(document["snapshot_set_id"]),
            screening_result_id=str(document["screening_result_id"]),
            compressed_belief_id=str(document["compressed_belief_id"]),
            workflow_runtime_s=float(document["workflow_runtime_s"]),
            total_executables=int(document["total_executables"]),
            pre_screen_selected=int(document["pre_screen_selected"]),
            probe_selected=int(document["probe_selected"]),
            survivor_selected=int(document["survivor_selected"]),
            persisted_signals=int(document["persisted_signals"]),
            evaluation_inputs=int(document["evaluation_inputs"]),
            pruned_snapshots=int(document["pruned_snapshots"]),
            created_at=str(document["created_at"]),
        )
